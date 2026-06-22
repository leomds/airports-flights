import os 
import geopandas as gpd 
from shapely .geometry import Point 

DATA_DIR =os .path .join (os .path .dirname (os .path .dirname (__file__ )),"data")

CONTINENTS =[
"America do Norte","America Central e Caribe",
"America do Sul","Europa","Africa","Asia","Oceania",
]
BR_REGIONS =["Norte","Nordeste","Centro-Oeste","Sudeste","Sul"]

_NE_CONTINENT_MAP ={
"North America":"America do Norte",
"South America":"America do Sul",
"Europe":"Europa",
"Africa":"Africa",
"Asia":"Asia",
"Oceania":"Oceania",
"Seven seas (open ocean)":"Oceania",
}

_BR_REGION_ID ={1 :"Sul",2 :"Sudeste",3 :"Norte",4 :"Nordeste",5 :"Centro-Oeste"}

_countries_gdf =None 
_states_gdf =None 

def _get_geodata ():
    global _countries_gdf ,_states_gdf 
    if _countries_gdf is None :
        c =gpd .read_file (os .path .join (DATA_DIR ,"countries.geojson"))
        c ["continent_pt"]=c ["CONTINENT"].map (_NE_CONTINENT_MAP ).fillna ("Outros")
        mask_ca =c ["SUBREGION"].isin (["Central America","Caribbean"])
        c .loc [mask_ca ,"continent_pt"]="America Central e Caribe"
        _countries_gdf =c 
        s =gpd .read_file (os .path .join (DATA_DIR ,"brazil_states.geojson"))
        s ["region"]=s ["regiao_id"].astype (int ).map (_BR_REGION_ID )
        _states_gdf =s 
    return _countries_gdf ,_states_gdf 


def _point_in_gdf (lon ,lat ,gdf ,col ):
    pt =Point (lon ,lat )
    for _ ,row in gdf .iterrows ():
        if row .geometry .contains (pt ):
            return row [col ]
    dists =gdf .geometry .distance (pt )
    return gdf .iloc [dists .idxmin ()][col ]


# gerado por ia
def classify_graph(G) -> dict:
    import pickle
    cache_path = os.path.join(DATA_DIR, "_geo_cache.pkl")
    # chave de cache: se o grafo não mudou, reutiliza sem recalcular
    cache_key = f"{G.number_of_nodes()}_{G.number_of_edges()}"

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            stored_key, result = pickle.load(f)
        # só usa o cache se o grafo for o mesmo
        if stored_key == cache_key:
            return result

    result = _classify_graph_impl(G)
    # salva em disco para reutilizar na próxima sessão
    with open(cache_path, "wb") as f:
        pickle.dump((cache_key, result), f)
    return result


def _classify_graph_impl (G )->dict :
    countries_gdf ,states_gdf =_get_geodata ()
    result ={
    "continent":{c :[]for c in CONTINENTS +["Outros"]},
    "country":{},
    "br_region":{r :[]for r in BR_REGIONS },
    "node_continent":{},
    "node_br_region":{},
    }
    for iata ,data in G .nodes (data =True ):
        country =data .get ("country","")
        lat =data .get ("lat")
        lon =data .get ("lon")
        continent =_point_in_gdf (lon ,lat ,countries_gdf ,"continent_pt")if lat and lon else "Outros"
        if not isinstance (continent ,str ):
            continent ="Outros"
        result ["node_continent"][iata ]=continent 
        result ["continent"].setdefault (continent ,[]).append (iata )
        result ["country"].setdefault (country ,[]).append (iata )
        if country =="Brazil"and lat and lon :
            region =_point_in_gdf (lon ,lat ,states_gdf ,"region")
            if not isinstance (region ,str ):
                region ="Norte"
            result ["node_br_region"][iata ]=region 
            result ["br_region"].setdefault (region ,[]).append (iata )
    return result 
