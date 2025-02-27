import logging
import random
import re
import numpy as np
import tiktoken
import umap
import copy
import asyncio
from abc import ABC, abstractmethod
from typing import List, Optional
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
from collections import Counter, defaultdict
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage
)
from ._utils import split_string_by_multi_markers, clean_str, is_float_regex
from .prompt import GRAPH_FIELD_SEP, PROMPTS

# Initialize logging
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

# Set a random seed for reproducibility
RANDOM_SEED = 224
random.seed(RANDOM_SEED)


def global_cluster_embeddings(
    embeddings: np.ndarray,
    dim: int,
    n_neighbors: int = 15,
    metric: str = "cosine",
) -> np.ndarray:
    if n_neighbors is None:
        n_neighbors = int((len(embeddings) - 1) ** 0.5)
    reduced_embeddings = umap.UMAP(
        n_neighbors=n_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)
    return reduced_embeddings


def local_cluster_embeddings(
    embeddings: np.ndarray, dim: int, num_neighbors: int = 10, metric: str = "cosine"
) -> np.ndarray:
    reduced_embeddings = umap.UMAP(
        n_neighbors=num_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)
    return reduced_embeddings


def fit_gaussian_mixture(n_components, embeddings, random_state):
    gm = GaussianMixture(
        n_components=n_components,
        random_state=random_state,
        n_init=5,
        init_params='k-means++'
        )
    gm.fit(embeddings)
    return gm.bic(embeddings)


def get_optimal_clusters(embeddings, max_clusters=50, random_state=0, rel_tol=1e-3):
    max_clusters = min(len(embeddings), max_clusters)
    n_clusters = np.arange(1, max_clusters)
    bics = []
    prev_bic = float('inf')
    for n in tqdm(n_clusters):
        bic = fit_gaussian_mixture(n, embeddings, random_state)
        # print(bic)
        bics.append(bic)
        # early stop
        if (abs(prev_bic - bic) / abs(prev_bic)) < rel_tol:
            break
        prev_bic = bic
    optimal_clusters = n_clusters[np.argmin(bics)]
    return optimal_clusters


def GMM_cluster(embeddings: np.ndarray, threshold: float, random_state: int = 0):
    n_clusters = get_optimal_clusters(embeddings)
    gm = GaussianMixture(
        n_components=n_clusters, 
        random_state=random_state, 
        n_init=5,
        init_params='k-means++')
    gm.fit(embeddings)
    probs = gm.predict_proba(embeddings)        # [num, cluster_num]
    labels = [np.where(prob > threshold)[0] for prob in probs]
    return labels, n_clusters


def perform_clustering(
    embeddings: np.ndarray, dim: int, threshold: float, verbose: bool = False
) -> List[np.ndarray]:
    reduced_embeddings_global = global_cluster_embeddings(embeddings, min(dim, len(embeddings) -2))
    global_clusters, n_global_clusters = GMM_cluster(     # (num, 2)
        reduced_embeddings_global, threshold
    )

    if verbose:
        logging.info(f"Global Clusters: {n_global_clusters}")

    all_clusters = [[] for _ in range(len(embeddings))]
    embedding_to_index = {tuple(embedding): idx for idx, embedding in enumerate(embeddings)}
    for i in tqdm(range(n_global_clusters)):
        global_cluster_embeddings_ = embeddings[
            np.array([i in gc for gc in global_clusters])
        ]
        if verbose:
            logging.info(
                f"Nodes in Global Cluster {i}: {len(global_cluster_embeddings_)}"
            )
        if len(global_cluster_embeddings_) == 0:
            continue

        # embedding indices
        indices = [
            embedding_to_index[tuple(embedding)]
            for embedding in global_cluster_embeddings_
        ]

        # update
        for idx in indices:
            all_clusters[idx].append(i)

    all_clusters = [np.array(cluster) for cluster in all_clusters]

    if verbose:
        logging.info(f"Total Clusters: {len(n_global_clusters)}")
    return all_clusters


async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 4 or record_attributes[0] != '"entity"':
        return None
    # add this record as a node in the G
    entity_name = clean_str(record_attributes[1].upper())
    if not entity_name.strip():
        return None
    entity_type = clean_str(record_attributes[2].upper())
    entity_description = clean_str(record_attributes[3])
    entity_source_id = chunk_key
    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        source_id=entity_source_id,
    )


async def _handle_single_relationship_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 5 or record_attributes[0] != '"relationship"':
        return None
    # add this record as edge
    source = clean_str(record_attributes[1].upper())
    target = clean_str(record_attributes[2].upper())
    edge_description = clean_str(record_attributes[3])
    edge_source_id = chunk_key
    weight = (
        float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0
    )
    return dict(
        src_id=source,
        tgt_id=target,
        weight=weight,
        description=edge_description,
        source_id=edge_source_id,
    )


class ClusteringAlgorithm(ABC):
    @abstractmethod
    def perform_clustering(self, embeddings: np.ndarray, **kwargs) -> List[List[int]]:
        pass


class Hierarchical_Clustering(ClusteringAlgorithm):
    async def perform_clustering(
        self,
        entity_vdb: BaseVectorStorage,
        global_config: dict,
        entities: dict,
        layers: int = 50,
        max_length_in_cluster: int = 60000,
        tokenizer=tiktoken.get_encoding("cl100k_base"),
        reduction_dimension: int = 2,
        cluster_threshold: float = 0.1,
        verbose: bool = False,
        threshold: float = 0.98, # 0.99
        thredshold_change_rate: float = 0.05
    ) -> List[dict]:
        use_llm_func: callable = global_config["best_model_func"]
        # Get the embeddings from the nodes
        nodes = list(entities.values())
        embeddings = np.array([x["embedding"] for x in nodes])
        
        hierarchical_clusters = [nodes]
        pre_cluster_sparsity = 0.01
        for layer in range(layers):
            logging.info(f"############ Layer[{layer}] Clustering ############")
            # Perform the clustering
            clusters = perform_clustering(
                embeddings, dim=reduction_dimension, threshold=cluster_threshold
            )
            # Initialize an empty list to store the clusters of nodes
            node_clusters = []
            # Iterate over each unique label in the clusters
            unique_clusters = np.unique(np.concatenate(clusters))
            logging.info(f"[Clustered Label Num: {len(unique_clusters)} / Last Layer Total Entity Num: {len(nodes)}]")
            # calculate the number of nodes belong to each cluster
            cluster_sizes = Counter(np.concatenate(clusters))
            # calculate cluster sparsity
            cluster_sparsity = 1 - sum([x * (x - 1) for x in cluster_sizes.values()])/(len(nodes) * (len(nodes) - 1))
            cluster_sparsity_change_rate = (abs(cluster_sparsity - pre_cluster_sparsity) / pre_cluster_sparsity)
            pre_cluster_sparsity = cluster_sparsity
            logging.info(f"[Cluster Sparsity: {round(cluster_sparsity, 4) * 100}%]")
            # stop if there will be no improvements on clustering
            if cluster_sparsity >= threshold:
                logging.info(f"[Stop Clustering at Layer{layer} with Cluster Sparsity {cluster_sparsity}]")
                break
            if cluster_sparsity_change_rate <= thredshold_change_rate:
                logging.info(f"[Stop Clustering at Layer{layer} with Cluster Sparsity Change Rate {round(cluster_sparsity_change_rate, 4) * 100}%]")
                break
            # summarize
            for label in unique_clusters:
                # Get the indices of the nodes that belong to this cluster
                indices = [i for i, cluster in enumerate(clusters) if label in cluster]
                # Add the corresponding nodes to the node_clusters list
                cluster_nodes = [nodes[i] for i in indices]
                # Base case: if the cluster only has one node, do not attempt to recluster it
                logging.info(f"[Label{str(int(label))} Size: {len(cluster_nodes)}]")
                if len(cluster_nodes) == 1:
                    node_clusters += cluster_nodes
                    continue
                # Calculate the total length of the text in the nodes
                total_length = sum(
                    [len(tokenizer.encode(node["description"])) + len(tokenizer.encode(node["entity_name"])) for node in cluster_nodes]
                )
                base_discount = 0.8
                discount_times = 0
                # If the total length exceeds the maximum allowed length, reduce the node size
                while total_length > max_length_in_cluster:
                    logging.info(
                        f"Reducing cluster size with {base_discount * 100 * (base_discount**discount_times):.2f}% of entities"
                    )

                    # for node in cluster_nodes:
                    #     description = node["description"]
                    #     node['description'] = description[:int(len(description) * base_discount)]
                    
                    # Randomly select 80% of the nodes
                    num_to_select = max(1, int(len(cluster_nodes) * base_discount))  # Ensure at least one node is selected
                    cluster_nodes = random.sample(cluster_nodes, num_to_select)

                    # Recalculate the total length
                    total_length = sum(
                        [len(tokenizer.encode(node["description"])) + len(tokenizer.encode(node["entity_name"])) for node in cluster_nodes]
                    )
                    discount_times += 1
                # summarize and generate new entities
                entity_description_list = [f"({x['entity_name']}, {x['description']})" for x in cluster_nodes]
                context_base_summarize = dict(
                    tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
                    record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
                    completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
                    meta_attribute_list=PROMPTS["META_ENTITY_TYPES"],
                    entity_description_list=",".join(entity_description_list)
                    )
                summarize_prompt = PROMPTS["summary_clusters"]
                hint_prompt = summarize_prompt.format(**context_base_summarize)
                summarize_result = await use_llm_func(hint_prompt)
                chunk_key = ""
                # resolve results
                records = split_string_by_multi_markers(                                            # split entities from result --> list of entities
                    summarize_result,
                    [context_base_summarize["record_delimiter"], context_base_summarize["completion_delimiter"]],
                )
                maybe_nodes = defaultdict(list)
                maybe_edges = defaultdict(list)
                for record in records:
                    record = re.search(r"\((.*)\)", record)
                    if record is None:
                        continue
                    record = record.group(1)
                    record_attributes = split_string_by_multi_markers(          # split entity
                        record, [context_base_summarize["tuple_delimiter"]]
                    )
                    if_entities = await _handle_single_entity_extraction(       # get the name, type, desc, source_id of entity--> dict
                        record_attributes, chunk_key
                    )
                    if if_entities is not None:
                        maybe_nodes[if_entities["entity_name"]].append(if_entities)
                        continue

                    if_relation = await _handle_single_relationship_extraction(
                        record_attributes, chunk_key
                    )
                    if if_relation is not None:
                        maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(
                                if_relation
                        )
                # fetch all entities from results
                entity_results = (dict(maybe_nodes), dict(maybe_edges))
                all_entities_relations = {}
                for item in entity_results:
                    for k, v in item.items():
                        value = v[0]
                        all_entities_relations[k] = v[0]
                # fetch embeddings
                entity_discriptions = [v["description"] for k, v in all_entities_relations.items()]
                entity_sequence_embeddings = []
                embeddings_batch_size = 64
                num_embeddings_batches = (len(entity_discriptions) + embeddings_batch_size - 1) // embeddings_batch_size
                for i in range(num_embeddings_batches):
                    start_index = i * embeddings_batch_size
                    end_index = min((i + 1) * embeddings_batch_size, len(entity_discriptions))
                    batch = entity_discriptions[start_index:end_index]
                    result = await entity_vdb.embedding_func(batch)
                    entity_sequence_embeddings.extend(result)
                entity_embeddings = entity_sequence_embeddings
                for (k, v), x in zip(all_entities_relations.items(), entity_embeddings):
                    value = v
                    value["embedding"] = x
                    all_entities_relations[k] = value
                # append the attribute entities of current clustered set to results
                all_entities_relations = [v for k, v in all_entities_relations.items()]
                node_clusters += all_entities_relations
            hierarchical_clusters.append(node_clusters)
            # update nodes to be clustered in the next layer
            nodes = copy.deepcopy([x for x in node_clusters if "entity_name" in x.keys()])
            # filter the duplicate entities
            seen = set()        
            unique_nodes = []
            for item in nodes:
                entity_name = item['entity_name']
                if entity_name not in seen:
                    seen.add(entity_name)
                    unique_nodes.append(item)
            nodes = unique_nodes
            embeddings = np.array([x["embedding"] for x in unique_nodes])
            # stop if the number of deduplicated cluster is too small
            if len(embeddings) <= 2:
                logging.info(f"[Stop Clustering at Layer{layer} with entity num {len(embeddings)}]")
                break
        return hierarchical_clusters