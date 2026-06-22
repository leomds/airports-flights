import networkx as nx 
import numpy as np 
from typing import Optional 


def compute_degree_centrality (G :nx .DiGraph )->dict :
    return nx .degree_centrality (G )

def compute_in_degree_centrality (G :nx .DiGraph )->dict :
    return nx .in_degree_centrality (G )

def compute_out_degree_centrality (G :nx .DiGraph )->dict :
    return nx .out_degree_centrality (G )

def compute_betweenness_centrality (G :nx .DiGraph ,k :int =500 )->dict :
    return nx .betweenness_centrality (G ,k =k ,weight ="weight",normalized =True )

def compute_closeness_centrality (G :nx .DiGraph )->dict :
    return nx .closeness_centrality (G ,distance ="weight")

def compute_pagerank (G :nx .DiGraph ,alpha :float =0.85 )->dict :
    return nx .pagerank (G ,alpha =alpha ,weight ="weight",max_iter =200 )

def top_nodes (metric :dict ,n :int =15 )->list :
    return sorted (metric .items (),key =lambda x :x [1 ],reverse =True )[:n ]


def get_weakly_connected_components (G :nx .DiGraph ):
    return list (nx .weakly_connected_components (G ))

def get_strongly_connected_components (G :nx .DiGraph ):
    return list (nx .strongly_connected_components (G ))

def get_bridges (G :nx .DiGraph ):
    U =G .to_undirected ()
    return list (nx .bridges (U ))

def get_articulation_points (G :nx .DiGraph ):
    U =G .to_undirected ()
    return list (nx .articulation_points (U ))

def connectivity_summary (G :nx .DiGraph )->dict :
    wcc =get_weakly_connected_components (G )
    scc =get_strongly_connected_components (G )
    bridges =get_bridges (G )
    artics =get_articulation_points (G )
    degrees =[d for _ ,d in G .degree ()]

    return {
    "nodes":G .number_of_nodes (),
    "edges":G .number_of_edges (),
    "weakly_connected_components":len (wcc ),
    "largest_wcc_size":max (len (c )for c in wcc ),
    "strongly_connected_components":len (scc ),
    "largest_scc_size":max (len (c )for c in scc ),
    "bridges":len (bridges ),
    "articulation_points":len (artics ),
    "density":nx .density (G ),
    "avg_degree":np .mean (degrees ),
    "max_degree":int (np .max (degrees )),
    "median_degree":float (np .median (degrees )),
    }


# gerado por ia
def degree_distribution(G: nx.DiGraph) -> dict:
    degrees = [d for _, d in G.degree()]
    unique_k, counts_raw = np.unique(degrees, return_counts=True)
    # normaliza as contagens para obter probabilidade P(k)
    pk = counts_raw / counts_raw.sum()

    # exclui k=0 porque log(0) é indefinido
    mask = (unique_k > 0) & (pk > 0)
    log_k = np.log10(unique_k[mask].astype(float))
    log_pk = np.log10(pk[mask])
    # regressão linear no espaço log-log: coeficiente angular é -alpha
    coeffs = np.polyfit(log_k, log_pk, 1)
    alpha = -coeffs[0]

    # calcula R² para medir a qualidade do ajuste
    fitted = np.polyval(coeffs, log_k)
    ss_res = np.sum((log_pk - fitted) ** 2)
    ss_tot = np.sum((log_pk - log_pk.mean()) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "degrees": degrees,
        "bins": unique_k.tolist(),
        "pk": pk.tolist(),
        "alpha": float(alpha),
        "r_squared": float(r_squared),
    }


def average_clustering_coefficient (G :nx .DiGraph )->float :
    U =G .to_undirected ()
    return nx .average_clustering (U )


def average_shortest_path_sample (G :nx .DiGraph ,sample :int =500 ,seed :int =42 )->float :
    import random 
    rng =random .Random (seed )
    wcc =max (nx .weakly_connected_components (G ),key =len )
    subG =G .subgraph (wcc ).copy ()
    nodes =list (subG .nodes ())
    if len (nodes )<2 :
        return 0.0 
    total ,count =0.0 ,0 
    for _ in range (sample ):
        src ,dst =rng .sample (nodes ,2 )
        try :
            length =nx .shortest_path_length (subG ,src ,dst )
            total +=length 
            count +=1 
        except nx .NetworkXNoPath :
            pass 
    return total /count if count else 0.0 


def shortest_path (G :nx .DiGraph ,src :str ,dst :str ,weight :str ="weight"):
    try :
        path =nx .dijkstra_path (G ,src ,dst ,weight =weight )
        length =nx .dijkstra_path_length (G ,src ,dst ,weight =weight )
        return path ,length 
    except (nx .NetworkXNoPath ,nx .NodeNotFound ):
        return None ,None 

def shortest_path_avoiding (G :nx .DiGraph ,src :str ,dst :str ,
avoid_nodes :list =None ,avoid_edges :list =None ,
weight :str ="weight"):
    H =G .copy ()
    if avoid_nodes :
        H .remove_nodes_from ([n for n in avoid_nodes if n in H and n !=src and n !=dst ])
    if avoid_edges :
        H .remove_edges_from ([e for e in avoid_edges if H .has_edge (*e )])
    return shortest_path (H ,src ,dst ,weight =weight )

def path_edges (path :list )->list :
    return list (zip (path [:-1 ],path [1 :]))

def path_total_distance (G :nx .DiGraph ,path :list )->float :
    total =0.0 
    for u ,v in path_edges (path ):
        total +=G [u ][v ].get ("weight",0 )
    return total 


def simulate_node_removal (G :nx .DiGraph ,nodes_to_remove :list )->dict :
    H =G .copy ()
    original_edges =G .number_of_edges ()
    original_wcc =max (len (c )for c in nx .weakly_connected_components (G ))
    H .remove_nodes_from ([n for n in nodes_to_remove if n in H ])
    new_edges =H .number_of_edges ()
    if H .number_of_nodes ()>0 :
        new_wcc =max (len (c )for c in nx .weakly_connected_components (H ))
        new_avg_deg =np .mean ([d for _ ,d in H .degree ()])
    else :
        new_wcc =0 
        new_avg_deg =0.0 
    return {
    "removed_nodes":len (nodes_to_remove ),
    "original_nodes":G .number_of_nodes (),
    "remaining_nodes":H .number_of_nodes (),
    "original_edges":original_edges ,
    "remaining_edges":new_edges ,
    "edges_lost_pct":(original_edges -new_edges )/original_edges *100 ,
    "original_largest_wcc":original_wcc ,
    "new_largest_wcc":new_wcc ,
    "wcc_reduction_pct":(original_wcc -new_wcc )/original_wcc *100 ,
    "new_avg_degree":new_avg_deg ,
    }

def robustness_curve (G :nx .DiGraph ,strategy :str ="degree",steps :int =30 )->list :
    H =G .copy ()
    N =H .number_of_nodes ()
    results =[(0.0 ,1.0 )]
    batch =max (1 ,N //steps )
    for step in range (steps ):
        if H .number_of_nodes ()==0 :
            break 
        if strategy =="degree":
            ranking =sorted (H .degree (),key =lambda x :x [1 ],reverse =True )
        elif strategy =="betweenness":
            bc =nx .betweenness_centrality (H ,k =min (100 ,H .number_of_nodes ()))
            ranking =sorted (bc .items (),key =lambda x :x [1 ],reverse =True )
        else :
            import random 
            nodes =list (H .nodes ())
            random .shuffle (nodes )
            ranking =[(n ,0 )for n in nodes ]
        to_remove =[n for n ,_ in ranking [:batch ]]
        H .remove_nodes_from (to_remove )
        if H .number_of_nodes ()==0 :
            results .append (((step +1 )*batch /N ,0.0 ))
            break 
        largest =max (len (c )for c in nx .weakly_connected_components (H ))
        results .append (((step +1 )*batch /N ,largest /N ))
    return results 
