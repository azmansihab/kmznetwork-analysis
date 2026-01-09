import streamlit as st
import geopandas as gpd
import pandas as pd
import osmnx as ox
import networkx as nx
import simplekml
import zipfile
import numpy as np
import os
from scipy.spatial import ConvexHull
from shapely.geometry import Point

# Konfigurasi Cache OSMNX agar tidak download berulang kali
ox.config(use_cache=True, log_console=False)

def extract_kml_from_kmz(kmz_file):
    with zipfile.ZipFile(kmz_file, 'r') as z:
        kml_filename = [f for f in z.namelist() if f.endswith('.kml')][0]
        with z.open(kml_filename) as f:
            return f.read()

def load_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.kmz'):
            kml_data = extract_kml_from_kmz(uploaded_file)
            temp_name = f"temp_{uploaded_file.name.split('.')[0]}.kml"
            with open(temp_name, "wb") as f:
                f.write(kml_data)
            gdf = gpd.read_file(temp_name, driver='KML')
            if os.path.exists(temp_name): os.remove(temp_name)
        else:
            gdf = gpd.read_file(uploaded_file, driver='KML')
        return gdf
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        return None

st.set_page_config(page_title="ISP Network Analysis Pro", layout="wide")
st.title("🌐 ISP Network & Road-Based Analysis")
st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    file_tiang = st.file_uploader("1. Upload KMZ/KML Tiang", type=['kml', 'kmz'])
with col2:
    file_rumah = st.file_uploader("2. Upload KMZ/KML Rumah", type=['kml', 'kmz'])

if file_tiang and file_rumah:
    gdf_tiang = load_data(file_tiang)
    gdf_rumah = load_data(file_rumah)
    
    if gdf_tiang is not None and gdf_rumah is not None:
        st.success(f"Terbaca {len(gdf_tiang)} Tiang & {len(gdf_rumah)} Rumah")

        if st.button("🚀 Mulai Analisis Jalur & Boundary"):
            try:
                with st.spinner("Mengunduh data jalan & memproses rute..."):
                    # Ambil titik tengah area
                    avg_lat = gdf_tiang.geometry.y.mean()
                    avg_lon = gdf_tiang.geometry.x.mean()
                    
                    # Download Graph Jalan
                    G = ox.graph_from_point((avg_lat, avg_lon), dist=3000, network_type='drive')
                    
                    kml_out = simplekml.Kml()
                    colors = [simplekml.Color.red, simplekml.Color.blue, simplekml.Color.green, 
                              simplekml.Color.yellow, simplekml.Color.purple, simplekml.Color.orange]

                    for i, tiang in gdf_tiang.iterrows():
                        kode_tiang = tiang.get('Name', f"T-{i+1}")
                        tiang_coord = (tiang.geometry.y, tiang.geometry.x)
                        tiang_node = ox.distance.nearest_nodes(G, tiang_coord[1], tiang_coord[0])
                        
                        fol = kml_out.newfolder(name=f"AREA_{kode_tiang}")
                        
                        # Titik Tiang
                        p_tiang = fol.newpoint(name=f"PUSAT_{kode_tiang}", coords=[(tiang.geometry.x, tiang.geometry.y)])
                        p_tiang.style.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/shapes/target.png'

                        valid_homes = []
                        boundary_points = [(tiang.geometry.x, tiang.geometry.y)]

                        for j, rumah in gdf_rumah.iterrows():
                            r_coord = (rumah.geometry.y, rumah.geometry.x)
                            r_node = ox.distance.nearest_nodes(G, r_coord[1], r_coord[0])
                            
                            try:
                                dist = nx.shortest_path_length(G, tiang_node, r_node, weight='length')
                                if dist <= 250:
                                    path = nx.shortest_path(G, tiang_node, r_node, weight='length')
                                    l_coords = [(G.nodes[n]['x'], G.nodes[n]['y']) for n in path]
                                    valid_homes.append({'dist': dist, 'coords': (rumah.geometry.x, rumah.geometry.y), 'path': l_coords, 'name': rumah.get('Name', f"R-{j}")})
                            except:
                                continue

                        # Ambil 10 terdekat
                        selected_homes = sorted(valid_homes, key=lambda x: x['dist'])[:10]

                        for home in selected_homes:
                            boundary_points.append(home['coords'])
                            # Titik Rumah
                            pr = fol.newpoint(name=f"{kode_tiang}_{home['name']} ({int(home['dist'])}m)", coords=[home['coords']])
                            pr.style.iconstyle.scale = 0.8
                            # Garis Jalan
                            ls = fol.newlinestring(name=f"Jalur {home['name']}")
                            ls.coords = home['path']
                            ls.style.linestyle.width = 3
                            ls.style.linestyle.color = colors[i % len(colors)]

                        # Pembuatan Boundary
                        if len(boundary_points) >= 3:
                            pts_arr = np.array(boundary_points)
                            hull = ConvexHull(pts_arr)
                            hull_pts = np.vstack([pts_arr[hull.vertices], pts_arr[hull.vertices[0]]])
                            
                            poly = fol.newpolygon(name=f"BOUNDARY_{kode_tiang}")
                            poly.outerboundaryis = [(p[0], p[1]) for p in hull_pts]
                            poly.style.polystyle.color = simplekml.Color.changealphaint(50, colors[i % len(colors)])
                            poly.style.linestyle.color = colors[i % len(colors)]

                    output_file = "ISP_Final_Analysis.kml"
                    kml_out.save(output_file)
                    st.success("✅ Analisis Selesai!")
                    
                    with open(output_file, "rb") as f:
                        st.download_button("📥 Download Hasil KMZ", f, file_name="Hasil_ISP_Network.kmz")

            except Exception as outer_e:
                st.error(f"Terjadi kesalahan teknis: {outer_e}")
else:
    st.info("Silakan unggah kedua file KMZ/KML untuk memulai.")