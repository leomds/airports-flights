import folium 
import networkx as nx 
import math 
import random 


def _arc_points (lat1 ,lon1 ,lat2 ,lon2 ,n =30 ):
    la1 ,lo1 =math .radians (lat1 ),math .radians (lon1 )
    la2 ,lo2 =math .radians (lat2 ),math .radians (lon2 )
    d =2 *math .asin (math .sqrt (
    math .sin ((la2 -la1 )/2 )**2 +
    math .cos (la1 )*math .cos (la2 )*math .sin ((lo2 -lo1 )/2 )**2 ))
    if d <1e-10 :
        return [(lat1 ,lon1 ),(lat2 ,lon2 )]
    points =[]
    for i in range (n +1 ):
        f =i /n 
        A =math .sin ((1 -f )*d )/math .sin (d )
        B =math .sin (f *d )/math .sin (d )
        x =A *math .cos (la1 )*math .cos (lo1 )+B *math .cos (la2 )*math .cos (lo2 )
        y =A *math .cos (la1 )*math .sin (lo1 )+B *math .cos (la2 )*math .sin (lo2 )
        z =A *math .sin (la1 )+B *math .sin (la2 )
        lat =math .degrees (math .atan2 (z ,math .sqrt (x **2 +y **2 )))
        lon =math .degrees (math .atan2 (y ,x ))
        points .append ((lat ,lon ))
    return points


# gerado por ia
def _split_antimeridian(points: list) -> list:
    if not points:
        return [points]
    segments = []
    current = [points[0]]
    for i in range(1, len(points)):
        prev_lon = points[i - 1][1]
        curr_lon = points[i][1]
        # salto de mais de 180 graus indica cruzamento do antimeridiano
        if abs(curr_lon - prev_lon) > 180:
            # interpola onde exatamente a rota toca a borda +-180
            frac = (180 - abs(prev_lon)) / (abs(curr_lon - prev_lon))
            lat_cross = points[i - 1][0] + frac * (points[i][0] - points[i - 1][0])
            border_lon = 180.0 if prev_lon > 0 else -180.0
            # fecha o segmento atual na borda
            current.append((lat_cross, border_lon))
            segments.append(current)
            # começa o próximo segmento no lado oposto da borda
            current = [(lat_cross, -border_lon), points[i]]
        else:
            current.append(points[i])
    segments.append(current)
    return segments


def build_base_map (center =(20 ,0 ),zoom =2 )->folium .Map :
    m =folium .Map (
    location =center ,
    zoom_start =zoom ,
    tiles =None ,
    prefer_canvas =True ,
    max_bounds =True ,
    min_zoom =2 ,
    )
    folium .TileLayer (
    tiles ="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
    attr ='&copy; <a href="https://carto.com/">CARTO</a>',
    name ="CartoDB Dark Matter",
    no_wrap =True ,
    subdomains ="abcd",
    ).add_to (m )
    m .fit_bounds ([[-85 ,-180 ],[85 ,180 ]])
    return m 


def add_airports_scaled (m :folium .Map ,G :nx .DiGraph ,
nodes :list =None ,highlight :list =None ):
    highlight =highlight or []
    target =nodes if nodes else list (G .nodes ())
    degrees =dict (G .degree ())
    max_deg =max (degrees .values ())if degrees else 1
    for iata in target :
        if iata not in G .nodes :
            continue 
        data =G .nodes [iata ]
        lat ,lon =data .get ("lat"),data .get ("lon")
        if lat is None or lon is None :
            continue 
        if iata in highlight :
            radius =8 
            color ="#ff6b35"
            opacity =1.0 
        else :
            deg =degrees .get (iata ,1 )
            norm =deg /max_deg 
            radius =2 +norm *9 
            r =int (0 +norm *0 )
            g =int (150 +norm *105 )
            b =int (180 +norm *75 )
            color =f"#{r :02x}{g :02x}{b :02x}"
            opacity =0.45 +norm *0.5 
        folium .CircleMarker (
        location =[lat ,lon ],
        radius =radius ,
        color =color ,
        fill =True ,
        fill_color =color ,
        fill_opacity =opacity ,
        weight =0 ,
        tooltip =(
        f"<b>{iata }</b> -- {data .get ('city','')}, {data .get ('country','')}<br>"
        f"{data .get ('name','')}<br>"
        f"Grau: {degrees .get (iata ,0 )}"
        ),
        ).add_to (m )
    return m 


def add_airports (m ,G ,nodes =None ,color ="#00d4ff",radius =3 ,highlight =None ):
    return add_airports_scaled (m ,G ,nodes =nodes ,highlight =highlight )


def add_route (m :folium .Map ,G :nx .DiGraph ,path :list ,
color ="#ff6b35",weight =3 ,label ="Rota"):
    for u ,v in zip (path [:-1 ],path [1 :]):
        if u not in G .nodes or v not in G .nodes :
            continue 
        n1 ,n2 =G .nodes [u ],G .nodes [v ]
        arc =_arc_points (n1 ["lat"],n1 ["lon"],n2 ["lat"],n2 ["lon"])
        for segment in _split_antimeridian (arc ):
            if len (segment )<2 :
                continue 
            folium .PolyLine (
            locations =segment ,
            color =color ,
            weight =weight ,
            opacity =0.9 ,
            tooltip =f"{u } -> {v }",
            ).add_to (m )
    return m 


def add_background_routes (m :folium .Map ,G :nx .DiGraph ,
sample_nodes :list =None ,max_edges :int =None ):
    drawn =set ()
    for u ,v in G .edges ():
        if u not in G .nodes or v not in G .nodes :
            continue 
        if sample_nodes and u not in sample_nodes and v not in sample_nodes :
            continue 
        key =(min (u ,v ),max (u ,v ))
        if key in drawn :
            continue 
        drawn .add (key )
        n1 ,n2 =G .nodes [u ],G .nodes [v ]
        folium .PolyLine (
        locations =[[n1 ["lat"],n1 ["lon"]],[n2 ["lat"],n2 ["lon"]]],
        color ="#3a6bbf",
        weight =0.8 ,
        opacity =0.3 ,
        ).add_to (m )
    return m 


def add_centrality_layer (m :folium .Map ,G :nx .DiGraph ,
centrality :dict ,top_n :int =50 ,
color_scale =None ):
    top =sorted (centrality .items (),key =lambda x :x [1 ],reverse =True )[:top_n ]
    max_val =top [0 ][1 ]if top else 1 
    min_val =top [-1 ][1 ]if top else 0 
    rng =max_val -min_val or 1 
    for rank ,(iata ,val )in enumerate (top ):
        if iata not in G .nodes :
            continue 
        data =G .nodes [iata ]
        lat ,lon =data .get ("lat"),data .get ("lon")
        if lat is None or lon is None :
            continue 
        norm =(val -min_val )/rng 
        radius =4 +norm *18 
        r =int (norm *255 )
        g =int ((1 -norm )*120 )
        b =int ((1 -norm )*200 )
        hex_color =f"#{r :02x}{g :02x}{b :02x}"
        folium .CircleMarker (
        location =[lat ,lon ],
        radius =radius ,
        color =hex_color ,
        fill =True ,
        fill_color =hex_color ,
        fill_opacity =0.85 ,
        weight =0 ,
        tooltip =(
        f"<b>{iata }</b> -- {data .get ('name','')}<br>"
        f"{data .get ('city','')}, {data .get ('country','')}<br>"
        f"Score: {val :.5f} | Rank #{rank +1 }"
        ),
        ).add_to (m )
    return m 


def add_removed_nodes (m :folium .Map ,G :nx .DiGraph ,nodes :list ):
    for iata in nodes :
        if iata not in G .nodes :
            continue 
        data =G .nodes [iata ]
        lat ,lon =data .get ("lat"),data .get ("lon")
        if lat is None or lon is None :
            continue 
        folium .Marker (
        location =[lat ,lon ],
        icon =folium .Icon (color ="red",icon ="times",prefix ="fa"),
        tooltip =f"X {iata } -- {data .get ('name','')} (indisponivel)",
        ).add_to (m )
    return m 


def map_to_html (m :folium .Map )->str :
    return m ._repr_html_ ()
