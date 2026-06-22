import pandas as pd 
import numpy as np 
import networkx as nx 
import math 
import os 

DATA_DIR =os .path .join (os .path .dirname (os .path .dirname (__file__ )),"data")

AIRPORT_COLS =[
"airport_id","name","city","country","iata","icao",
"lat","lon","altitude","timezone","dst","tz","type","source"
]

ROUTE_COLS =[
"airline","airline_id","src_iata","src_id",
"dst_iata","dst_id","codeshare","stops","equipment"
]


def load_airports()->pd .DataFrame :
    df =pd .read_csv (
    os .path .join (DATA_DIR ,"airports.dat"),
    header =None ,
    names =AIRPORT_COLS ,
    na_values =["\\N",""],
    low_memory =False ,
    )
    df =df [df ["iata"].notna ()&(df ["iata"].str .len ()==3 )]
    df =df [df ["lat"].notna ()&df ["lon"].notna ()]
    df ["lat"]=pd .to_numeric (df ["lat"],errors ="coerce")
    df ["lon"]=pd .to_numeric (df ["lon"],errors ="coerce")
    df =df .dropna (subset =["lat","lon"])
    return df .set_index ("iata")


def load_routes()->pd .DataFrame :
    df =pd .read_csv (
    os .path .join (DATA_DIR ,"routes.dat"),
    header =None ,
    names =ROUTE_COLS ,
    na_values =["\\N",""],
    low_memory =False ,
    )
    df =df [df ["stops"]==0 ]
    df =df [df ["src_iata"].notna ()&df ["dst_iata"].notna ()]
    df =df [df ["src_iata"].str .len ()==3 ]
    df =df [df ["dst_iata"].str .len ()==3 ]
    return df 


# gerado por ia
def haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0  # raio médio da Terra em km
    # converte as latitudes de graus para radianos
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    # calcula a diferença de latitude e longitude em radianos
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    # fórmula de haversine: combina as diferenças angulares
    # considerando a curvatura da esfera
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    # converte o ângulo resultante para distância em km
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def build_graph (airports :pd .DataFrame ,routes :pd .DataFrame )->nx .DiGraph :
    G =nx .DiGraph ()

    for iata ,row in airports .iterrows ():
        G .add_node (iata ,
        name =row ["name"],
        city =str (row .get ("city","")),
        country =str (row .get ("country","")),
        lat =float (row ["lat"]),
        lon =float (row ["lon"]))

    for _ ,row in routes .iterrows ():
        src ,dst =row ["src_iata"],row ["dst_iata"]
        if src in G .nodes and dst in G .nodes :
            src_data =G .nodes [src ]
            dst_data =G .nodes [dst ]
            dist =haversine (src_data ["lat"],src_data ["lon"],
            dst_data ["lat"],dst_data ["lon"])
            G .add_edge (src ,dst ,weight =dist ,airline =str (row .get ("airline","")))

    isolated =[n for n ,d in G .degree ()if d ==0 ]
    G .remove_nodes_from (isolated )
    print (f"[data_loader] Removidos {len (isolated )} nos isolados. "
    f"Rede final: {G .number_of_nodes ()} aeroportos, {G .number_of_edges ()} rotas.")

    return G 


def get_undirected (G :nx .DiGraph )->nx .Graph :
    return G .to_undirected ()


_graph_cache =None 
_airports_cache =None 


def get_data ():
    global _graph_cache ,_airports_cache 
    if _graph_cache is None :
        airports =load_airports ()
        routes =load_routes ()
        _airports_cache =airports 
        _graph_cache =build_graph (airports ,routes )
    return _airports_cache ,_graph_cache 
