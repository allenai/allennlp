from typing import Dict, List, Set
import numpy as np
import torch
from allennlp.interpret.attack import Attacker
from allennlp.common.util import JsonDict, sanitize 
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
from collections import defaultdict

@Attacker.register('input-reduction')
class InputReduction(Attacker):
    def __init__(self, predictor):
        super().__init__(predictor)

    def attack_from_json(self, inputs:JsonDict, target_field: str, gradient_index:str,ignore_tokens:List[str] = ["@@NULL@@"]):                        
        fields_to_check = set()
        fields_to_check_list = {}
        
        original_instances = self.predictor.inputs_to_labeled_instances(inputs)
        final_tokens = []
        for i in range(len(original_instances)):
            current_instances = [original_instances[i]]            
            original_tokens = [x for x in current_instances[0][target_field].tokens]
            grads,outputs = self.predictor.get_gradients(current_instances)
            
            test_instances = self.predictor.inputs_to_labeled_instances(inputs)
            for key in current_instances[0].fields.keys():
                # we want to check all model prediction fields
                if key not in inputs.keys() and key != target_field:
                    fields_to_check.add(key)
                    fields_to_check_list[key] = test_instances[0][key]

            # Keep at least one token for classification/entailment/etc.
            if "tags" not in current_instances[0]:
                num_ignore_tokens = 1 
            
            # Set num_ignore_tokens for NER and build token mask
            else:            
                # Set num_ignore_tokens
                num_ignore_tokens = 0
                for token in current_instances[0][target_field].tokens:                
                    if str(token) in ignore_tokens:
                        num_ignore_tokens += 1
                        
                tag_dict = defaultdict(int)
                tag_tok = ''
                og_mask = []
                og_tags = []
                for label in current_instances[0]["tags"].__dict__["field_list"]:                    
                    if label.label != "O":
                        tag_dict[label.label] += 1
                        tag_tok = tag_tok + label.label
                        og_mask.append(1)
                        og_tags.append(label.label)
                        num_ignore_tokens +=1
                    else:
                        og_mask.append(0)                

                
            current_tokens = current_instances[0][target_field].tokens
            idx = -1
            while len(current_instances[0][target_field]) >= num_ignore_tokens:
                grads, outputs = self.predictor.get_gradients(current_instances)                                
                for output in outputs:
                    if isinstance(outputs[output], torch.Tensor):
                        outputs[output] = outputs[output].detach().cpu().numpy().squeeze().squeeze()                         
                    elif isinstance(outputs[output],list):
                        outputs[output] = outputs[output][0]                                        
                                    
                # Check if any fields have changed, if so, break loop
                # special case for sentence tagging (we have tested NER)    
                if "tags"  in current_instances[0]:
                    if idx != -1:
                        del og_mask[idx]                
                    cur_tags = [outputs["tags"][x] for x in range(len(outputs["tags"])) if og_mask[x]]                                        
                    if cur_tags != og_tags:
                        break
                else:        
                    equal = True            
                    for field in fields_to_check:                        
                        if field in current_instances[0].fields:
                            equal = equal and current_instances[0][field].__eq__(fields_to_check_list[field])                            
                        else:
                            equal = equal and outputs[field] == fields_to_check_list[field]
                    if not equal:
                        break               
                current_tokens = list(current_instances[0][target_field].tokens)                                 
                current_instances = self.input_reduction(grads[gradient_index], current_instances, target_field, ignore_tokens)
                    
            final_tokens.append(current_tokens)
        return sanitize({"final": final_tokens,"original":original_tokens})
    
    def input_reduction(self, grads:np.ndarray, instances:List[Instance], target_field: str, ignore_tokens:List[str] = ["@@NULL@@"]) -> List[Instance]:     
        """
        Finds the token with the smallest gradient and removes it.        
        """        
        # Compute L2 norm of all grads. 
        grads_mag = [np.sqrt(grad.dot(grad)) for grad in grads]
        
        # Skip all ignore_tokens by setting grad to infinity
        for tok_idx, tok in enumerate(instances[0][target_field].tokens):
            if tok in ignore_tokens:
                grads_mag[tok_idx] = float("inf")
        # For NER, skip all tokens that are not in outside
        if "tags" in instances[0]:            
            for idx,label in enumerate(instances[0]["tags"].__dict__["field_list"]):
                if label.label != "O":
                    grads_mag[idx] = float("inf")

        smallest = np.argmin(grads_mag)        
        if smallest == float("inf"):
            return instances

        # remove smallest
        if "tags" in instances[0]:
            instances[0]["tags"].__dict__["field_list"] = instances[0]["tags"].__dict__["field_list"][0:smallest] + instances[0]["tags"].__dict__["field_list"][smallest + 1:]
        else:
            instances[0][target_field].tokens = instances[0][target_field].tokens[0:smallest] +  instances[0][target_field].tokens[smallest + 1:]
                            
        instances[0].indexed = False
        return instances