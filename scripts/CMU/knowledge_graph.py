import json
import re
import sys
from preprocess import Preprocessor

CONTEXT = "context"


class KnowledgeGraph(object):

    def __init__(self):
        self._preprocess = Preprocessor()

    def _update_line(self, line, graph, context_so_far):
        ent_pattern = r"\[[a-zA-Z0-9' ]+\([a-zA-Z]+\[\d+\]\)\]"
        ref_pattern = r"\[[a-zA-Z0-9' ]+\]"
        words = r"[a-zA-Z0-9' ]+"
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
                metadata = {'name': name, 'attributes': [], 'relation': {}, 'start_index': context_so_far.find(name)}
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
                        metadata['attributes'].append((attr, context_so_far.rfind(attr)))

    def prepare(self, path):
        graph = {}
        context_so_far = ""
        with open(path) as f:
            for line in f:
                if "question_" in line:
                    continue

                context_so_far += (" " if context_so_far else "") + self._preprocess._line_cleanup(line) + "."
                self._update_line(line, graph, context_so_far)

        graph[CONTEXT] = context_so_far
        return graph

    def prepare_edges(self, graph):
        nodes = []
        for key in graph:
            if key == CONTEXT:
                continue
            nodes.extend([(key + "-" + elem) for elem in graph[key]])

        sorted_nodes = sorted(nodes)
        nodes = {k: v for v, k in enumerate(sorted_nodes)}
        m_len = len(nodes)

        edges = []
        for _ in range(m_len):
            edges.append([-1] * m_len)

        print(json.dumps(nodes))
        for entry in nodes:
            class_type, id = entry.split("-")
            neighbors = graph[class_type][id]['relation']

            entry_index = nodes[entry]
            edges[entry_index][entry_index] = 0
            for neighbor_class, neighbor_nodes in neighbors.iteritems():
                for neighbor_node in neighbor_nodes:
                    neighbor = "%s-%s" % (neighbor_class, neighbor_node)
                    neighbor_index = nodes[neighbor]
                    edges[entry_index][neighbor_index] = 1
        return nodes, edges, sorted_nodes

class Dijkstra(object):

    def __init__(self, nodes, edges):
        self._nodes = nodes
        self._edges = edges

    def setupShortestPath(self, node, dist):
        print("setting up shortest path from", node)
        src_index = self._nodes[node]
        num_of_nodes = len(self._nodes)
        dist[src_index][src_index] = 0
        visited = set([src_index])
        next_index = src_index

        while len(visited) < num_of_nodes:
            # find all nodes connected by next_index

            neighbors = [neighbor_index for index in visited for neighbor_index, val in enumerate(self._edges[index]) if
                    val == 1 and neighbor_index not in visited]
            checked = []
            for neighbor_index in neighbors:
                if neighbor_index in visited:
                    print("Shouldn't come here")
                    continue

                src_neighbor = self._edges[src_index][neighbor_index]
                src_neighbor = src_neighbor if src_neighbor == 1 else dist[src_index][neighbor_index]

                src_next = self._edges[src_index][next_index]
                src_next = src_next if src_next == 1 else dist[src_index][next_index]

                next_neighbor = self._edges[next_index][neighbor_index]
                next_neighbor = next_neighbor if next_neighbor == 1 else dist[next_index][neighbor_index]

                dist[src_index][neighbor_index] = min(src_neighbor, src_next + next_neighbor)
                dist[neighbor_index][src_index] = dist[src_index][neighbor_index]
                checked.append((neighbor_index, dist[src_index][neighbor_index]))

            if checked:
                next_index, distance = sorted(checked, key=lambda x: x[1])[0]
                visited.add(next_index)
            else:
                break

    def shortestPath(self, sorted_nodes):
        node_len = len(self._nodes)
        dist = []
        for _ in range(node_len):
            dist.append([sys.maxsize] * node_len)

        for node in self._nodes:
            self.setupShortestPath(node, dist)

        for src in range(node_len):
            for dst in range(node_len):
                if dist[src][dst] != sys.maxsize:
                    print((sorted_nodes[src], sorted_nodes[dst], dist[src][dst]))



if __name__ == "__main__":
    kg = KnowledgeGraph()
    path = "/Users/prasoon/CMU/10608/pt-allennlp/scripts/CMU/meeting.with_hints"
    graph = kg.prepare(path)
    # print(json.dumps(graph))
    # sys.exit(0)
    nodes, edges, sorted_nodes = kg.prepare_edges(graph)
    dj = Dijkstra(nodes, edges)
    dj.shortestPath(sorted_nodes)
