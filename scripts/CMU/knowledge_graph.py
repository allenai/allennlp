import re


class KnowledgeGraph(object):

    def __init__(self):
        pass

    def _update_line(self, line, graph):
        ent_pattern = r"\[[a-zA-Z0-9 ]+\([a-zA-Z ]+\[\d+\]\)\]?"
        ref_pattern = r"\[[a-zA-Z0-9 ]+\]"
        words = r"[a-zA-Z0-9 ]+"
        entities = re.findall(ent_pattern, line)
        new_line = line

        for entity in entities:
            new_line = new_line.replace(entity, '')

        refs = re.findall(ref_pattern, new_line)
        for index, entity in enumerate(entities):
            name, class_type, identifier = re.findall(words, entity)
            # list of all objects that are of this entity
            graph_entity = graph.get(class_type)
            if graph_entity is None:
                graph_entity = {}
                graph[class_type] = graph_entity

            # check if identifier exists in the graph entity
            metadata = graph_entity.get(identifier)
            if metadata is None:
                metadata = {'name': name, 'attributes': [], 'relation': {}}
                graph_entity[identifier] = metadata

            other_entities = [ent for pos, ent in enumerate(entities) if index != pos]
            for ent in other_entities:
                ent_name, ent_class_type, ent_id = re.findall(words, ent)
                relation_ent = metadata['relation'].get(ent_class_type)
                if not relation_ent:
                    relation_ent = []
                    metadata['relation'][ent_class_type] = relation_ent
                if ent_id not in relation_ent:
                    relation_ent.append(ent_id)

            for ref in refs:
                attributes = re.findall(words, ref)
                for attr in attributes:
                    if attr not in metadata['attributes']:
                        metadata['attributes'].append(attr)

    def prepare(self, path):
        graph = {}
        with open(path) as f:
            for line in f:
                if "question_" in line:
                    continue

                self._update_line(line, graph)

        print(graph)


if __name__ == "__main__":
    kg = KnowledgeGraph()
    path = "/Users/prasoon/CMU/10608/pt-allennlp/scripts/CMU/student.with_hints"
    kg.prepare(path)
