from typing import Dict, List, Set
import numpy
import torch
from allennlp.interpret.attack import Attacker
from allennlp.common.util import JsonDict, sanitize 
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
from allennlp.data.fields.field import DataArray, Field
from allennlp.data.fields import IndexField
from allennlp.modules.text_field_embedders.text_field_embedder import TextFieldEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.data.tokenizers import Token
from collections import defaultdict

@Attacker.register('hotflip')
class Hotflip(Attacker):
    def __init__(self, predictor):
        super().__init__(predictor)        
        for module in self.predictor._model.modules():
            if isinstance(module, TextFieldEmbedder):
                M = module        
        
        vocab = self.predictor._model.vocab
        self.vocab = vocab
        all_tokens = list(vocab._token_to_index["tokens"].keys())

        a = [x for x in vocab._index_to_token["tokens"].keys()]
        b = torch.LongTensor(a)
        b = b.unsqueeze(0)
        c = {"tokens":b}                
        if "token_characters" in self.predictor._dataset_reader._token_indexers:
            tokenizer = self.predictor._dataset_reader._token_indexers["token_characters"]._character_tokenizer
            t = tokenizer.batch_tokenize(all_tokens)            
            tt = tokenizer.tokenize("How")            
            index = self.vocab.get_token_index("H", self.predictor._dataset_reader._token_indexers["token_characters"]._namespace)
            index2 = self.vocab.get_token_index("o", self.predictor._dataset_reader._token_indexers["token_characters"]._namespace)            

            character_tokens = []
            pad_length = max([len(x) for x in t])            
            if getattr(t[0][0], 'text_id', None) is not None:                
                for each in t:
                    tmp = [x.text_id for x in each]                    
                    tmp = tmp + [0] * (pad_length-len(tmp))
                    character_tokens.append(tmp)            
                character_tokens = torch.LongTensor(character_tokens)
                c["token_characters"] = character_tokens.unsqueeze(0)
            else:
                for each in t:
                    tmp = [self.vocab.get_token_index(x.text,self.predictor._dataset_reader._token_indexers["token_characters"]._namespace) for x in each]                    
                    tmp = tmp + [0] * (pad_length-len(tmp))
                    character_tokens.append(tmp)                
                character_tokens = torch.LongTensor(character_tokens)
                c["token_characters"] = character_tokens.unsqueeze(0)

            if "elmo" in self.predictor._dataset_reader._token_indexers:
                pad_length = pad_length + 2
                lltokens = []
                elmo_tokens = []
                for each in all_tokens:
                    ltokens = [Token(text=each)]                    
                    tmp = self.predictor._dataset_reader._token_indexers["elmo"].tokens_to_indices(ltokens, self.vocab,"sentence")
                    tmp = tmp["sentence"]                
                    lltokens.append(tmp[0])                
                lltokens = torch.LongTensor(lltokens)
                c["elmo"] = lltokens.unsqueeze(0)                
        embedding_matrix = M(c)
        embedding_matrix = embedding_matrix.squeeze()                
        self.token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'), embedding_dim=embedding_matrix.shape[1], weight=embedding_matrix,trainable=False)


    def attack_from_json(self, inputs:JsonDict, target_field: str, gradient_index:str):        
        og_instances = self.predictor.inputs_to_labeled_instances(inputs)        
        original = list(og_instances[0][target_field].tokens)        
        final_tokens = []        
        for i in range(len(og_instances)):            
            new_instances = [og_instances[i]]                        

            # handling fields that need to be checked            
            check_list = {}            
            test_instances = self.predictor.inputs_to_labeled_instances(inputs)
            for key in set(new_instances[0].fields.keys()):
                if key not in inputs.keys() and key != target_field:                    
                    check_list[key] = test_instances[0][key]            

            # Build label for NER
            if "tags" in new_instances[0]:                
                tag_dict = defaultdict(int)
                tag_tok = ''                
                for label in new_instances[0]["tags"].__dict__["field_list"]:                    
                    if label.label != "O":
                        tag_dict[label.label] += 1
                        tag_tok = tag_tok + label.label

            current_tokens = new_instances[0][target_field].tokens
            grads, outputs = self.predictor.get_gradients(new_instances)                    
            flipped = []
            while True:
                grad = grads[gradient_index]
                # Compute L2 norm of all grads. 
                grads_mag = [numpy.sqrt(g.dot(g)) for g in grad]

                if "tags" in new_instances[0]:                    
                    for idx, label in enumerate(new_instances[0]["tags"].__dict__["field_list"]):
                        if label.label != "O":
                            grads_mag[idx] = -1 # will never be flipped
                    for idx in flipped:
                        grads_mag[idx] = -1                
                
                token_to_flip = numpy.argmax(grads_mag)
                if grads_mag[token_to_flip] == -1:
                    break

                # Flip token
                flipped.append(token_to_flip)
                adv_token_idx = new_instances[0][target_field]._indexed_tokens["tokens"][token_to_flip]                           
                flip_token_id = self.first_order_taylor(grad[token_to_flip], self.token_embedding.weight, adv_token_idx).data[0].detach().cpu().item()                                                
                new_instances[0][target_field].tokens[token_to_flip] = Token(self.vocab._index_to_token["tokens"][flip_token_id])
                new_instances[0].indexed = False                

                grads, outputs = self.predictor.get_gradients(new_instances)                
                for each in outputs:
                    if isinstance(outputs[each], torch.Tensor):                        
                        outputs[each] = outputs[each].detach().cpu().numpy().squeeze().squeeze()
                    elif isinstance(outputs[each],list):                    
                        outputs[each] = outputs[each][0]                        
                self.predictor.predictions_to_labeled_instances(new_instances[0], outputs)                

                # if the prediction has changed, then stop
                if "tags" in new_instances[0]: # NER case                
                    cur_tag_dict = defaultdict(int)
                    cur_tag_tok = ''
                    for label in new_instances[0]["tags"].__dict__["field_list"]                    :
                        if label.label != "O":
                            cur_tag_dict[label.label] += 1
                            cur_tag_tok = cur_tag_tok + label.label                                        
                    if not (cur_tag_dict == tag_dict) and (cur_tag_tok == tag_tok):
                        break
                else:  
                    label_change = False                  
                    for field in check_list.keys():                        
                        if field in new_instances[0].fields:
                            equal = new_instances[0][field].__eq__(check_list[field])                            
                        else:
                            equal = outputs[field] == check_list[field]                            
                        # Save prediction                    
                        if not equal: 
                            label_change = True
                            break                                                   
                    if label_change:                    
                        if "probs" in outputs: # sentiment analysis
                            new_prediction = outputs["probs"]                    
                        elif "label_probs" in outputs: # textual entailment
                            new_prediction = outputs["label_probs"]                    
                        elif "best_span_str" in outputs: # bidaf
                            new_prediction = outputs["best_span_str"]
                        elif "answer" in outputs: # NAQANet
                            ans_type = outputs["answer"]["answer_type"]
                            if ans_type == "count":
                                new_prediction = outputs["answer"]["count"]
                            else:
                                new_prediction = outputs["answer"]["value"]
                        break
                        
            final_tokens.append(current_tokens)        
        return sanitize({"final": final_tokens,"original": original, "new_prediction": new_prediction})
        
    def first_order_taylor(self, grad, embedding_matrix, token_idx):            
        """
        The "Hotflip" attack described clearly in https://arxiv.org/abs/1903.06620, the below code is based on https://github.com/pmichel31415/translate/blob/paul/pytorch_translate/research/adversarial/adversaries/brute_force_adversary.py
        """
        grad = torch.from_numpy(grad)        
        embedding_matrix = embedding_matrix.cpu()    
        word_embeds = torch.nn.functional.embedding(torch.LongTensor([token_idx]), embedding_matrix).detach().unsqueeze(0)
        grad = grad.unsqueeze(0).unsqueeze(0)          
        new_embed_dot_grad = torch.einsum("bij,kj->bik", (grad, embedding_matrix))        
        prev_embed_dot_grad = torch.einsum("bij,bij->bi", (grad, word_embeds)).unsqueeze(-1)         
        neg_dir_dot_grad = -1 * (prev_embed_dot_grad - new_embed_dot_grad)            
        score_at_each_step, best_at_each_step = neg_dir_dot_grad.max(2)                    
        return best_at_each_step[0] # return the best candidate