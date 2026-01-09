import streamlit as st
import geopandas as gpd
import pandas as pd
import osmnx as ox
import networkx as nx
import simplekml
import zipfile
import numpy as np
import os
from scipy.spatial import ConvexHull, cKDTree

# Update OSMNX Settings
ox.settings.use_cache = True
ox.settings.log_console = False

def extract_kml_from_kmz(kmz_file):
    try:
        with zipfile.ZipFile(kmz_file, 'r') as z:
            kml_filename = [f for f in z.namelist() if f.endswith('.kml')][0]
            with z.open(kml_filename) as f:
                return f.read()
    except Exception as e:
        st.error(f"Gagal mengekstrak KMZ: {e}")
        return None

def load_data(uploaded_file):
    try:
        if uploaded_file.name.lower().endswith('.kmz'):
            kml_data = extract_kml_from_kmz(uploaded_file)
            if kml_data:
                temp_path = f"temp_{uploaded_file.name}.kml"
                with open(temp_path, "wb") as f:
                    f.write(kml_data)
                gdf = gpd.read_file(temp_path, driver='KML')
                if os.path.exists(temp_path): os.remove(temp_path)
                return gdf
        else:
            return gpd.read_file(uploaded_file, driver='KML')
    except Exception as e:
        st.error(f"Gagal membaca file {uploaded_file.name}: {e}")
        return None

def get_optimized_street_name(G, path, gdf_named_streets, r_coords):
    """Mencari nama jalan dengan performa tinggi."""
    names = []
    try:
        # 1. Coba ambil dari rute jalan yang dilewati
        gdf_route = ox.routing.route_to_gdf(G, path)
        if 'name' in gdf_route.columns:
            s_names = gdf_route['name'].dropna().tolist()
            for s in s_names:
                if isinstance(s, list): names.extend([str(x) for x in s])
                else: names.append(str(s))
    except: pass

    unique_names = sorted(list(set([n for n in names if n])))
    if unique_names:
        return ", ".join(unique_names)

    # 2. Fallback: Cari jalan bernama terdekat dari index spasial (Cepat)
    if not gdf_named_streets.empty:
        point = Point(r_coords[0], r_coords[1])
        # Cari jalan terdekat dalam GeoPandas (tanpa download ulang)
        dist = gdf_named_streets.distance(point)
        nearest_idx = dist.idxmin()
        return f"(Sekitar {gdf_named_streets.loc[nearest_idx, 'name']})"
    
    return "Jalan Lokal"

st.set_page_config(page_title="ISP Planner High-Performance", layout="wide")
st.title("🌐 ISP Network Planner: High Performance Analysis")

st.sidebar.header("⚙️ Parameter")
max_dist = st.sidebar.slider("Radius Maksimal Jalan (Meter)", 50, 1000, 250)
max_homes = st.sidebar.slider("Kapasitas Rumah per Tiang", 1, 32, 10)

col1, col2 = st.columns(2)
with col1:
    file_tiang = st.file_uploader("Upload KMZ Tiang", type=['kml', 'kmz'])
with col2:
    file_rumah = st.file_uploader("Upload KMZ Rumah", type=['kml', 'kmz'])

if file_tiang and file_rumah:
    gdf_tiang = load_data(file_tiang)
    gdf_rumah = load_data(file_rumah)

    if gdf_tiang is not None and gdf_rumah is not None:
        st.success(f"✅ Data Terdeteksi: {len(gdf_tiang)} Tiang & {len(gdf_rumah)} Rumah")
        
        if st.button("🚀 MULAI ANALISIS CEPAT"):
            with st.spinner("Proses optimasi sedang berjalan..."):
                # Hitung pusat area
                avg_lat, avg_lon = gdf_tiang.geometry.y.mean(), gdf_tiang.geometry.x.mean()
                
                # Download Graph & Fitur Jalan (Sekali saja di awal)
                G = ox.graph_from_point((avg_lat, avg_lon), dist=3000, network_type='all')
                
                # Ambil semua jalan yang punya nama di area tersebut untuk fallback cepat
                try:
                    gdf_named = ox.features_from_point((avg_lat, avg_lon), tags={'highway': True}, dist=3000)
                    gdf_named = gdf_named[gdf_named['name'].notna()][['name', 'geometry']]
                except:
                    gdf_named = gpd.GeoDataFrame()

                kml_out = simplekml.Kml()
                colors = [simplekml.Color.red, simplekml.Color.blue, simplekml.Color.green, 
                          simplekml.Color.yellow, simplekml.Color.purple, simplekml.Color.orange]

                all_connections = []
                tiang_list = []
                csv_data = [] 

                # 1. Alokasi Tiang
                for i, tiang in gdf_tiang.iterrows():
                    t_node = ox.distance.nearest_nodes(G, tiang.geometry.x, tiang.geometry.y)
                    tiang_list.append({'id': i, 'name': tiang.get('Name', f"T-{i+1}"), 'node': t_node, 'coords': (tiang.geometry.x, tiang.geometry.y)})

                # 2. Cari Semua Kemungkinan Jarak (Batch)
                for j, rumah in gdf_rumah.iterrows():
                    r_node = ox.distance.nearest_nodes(G, rumah.geometry.x, rumah.geometry.y)
                    r_coords = (rumah.geometry.x, rumah.geometry.y)
                    for t in tiang_list:
                        try:
                            dist = nx.shortest_path_length(G, t['node'], r_node, weight='length')
                            if dist <= max_dist:
                                all_connections.append({
                                    'tiang_idx': t['id'], 'rumah_idx': j, 'dist': dist,
                                    'r_coords': r_coords, 'r_name': rumah.get('Name', f"H-{j+1}"),
                                    'r_node': r_node
                                })
                        except: continue

                # 3. Optimasi Global
                all_connections = sorted(all_connections, key=lambda x: x['dist'])
                taken_homes = set()
                pole_load = {t['id']: 0 for t in tiang_list}
                final_allocations = {t['id']: [] for t in tiang_list}

                for conn in all_connections:
                    t_idx, r_idx = conn['tiang_idx'], conn['rumah_idx']
                    if r_idx not in taken_homes and pole_load[t_idx] < max_homes:
                        path = nx.shortest_path(G, tiang_list[t_idx]['node'], conn['r_node'], weight='length')
                        
                        # Ambil nama jalan dengan metode baru yang cepat
                        s_name = get_optimized_street_name(G, path, gdf_named, conn['r_coords'])
                        
                        conn['path'] = [(G.nodes[n]['x'], G.nodes[n]['y']) for n in path]
                        conn['streets'] = s_name
                        final_allocations[t_idx].append(conn)
                        taken_homes.add(r_idx)
                        pole_load[t_idx] += 1
                        
                        csv_data.append({
                            'KODE_TIANG': tiang_list[t_idx]['name'], 'NAMA_RUMAH': conn['r_name'],
                            'NAMA_JALAN': s_name, 'JARAK_KABEL_M': round(conn['dist'], 2),
                            'LONGITUDE': conn['r_coords'][0], 'LATITUDE': conn['r_coords'][1]
                        })

                # 4. Generate KMZ & Boundary
                for t in tiang_list:
                    t_idx = t['id']
                    fol = kml_out.newfolder(name=f"AREA_{t['name']}")
                    fol.newpoint(name=f"PUSAT_{t['name']}", coords=[t['coords']])
                    pts_boundary = [t['coords']]
                    for res in final_allocations[t_idx]:
                        pts_boundary.append(res['r_coords'])
                        p = fol.newpoint(name=res['r_name'], coords=[res['r_coords']])
                        p.description = f"Lokasi: {res['streets']}\nJarak: {int(res['dist'])}m"
                        ls = fol.newlinestring(name=f"Route {res['r_name']}", coords=res['path'])
                        ls.style.linestyle.color = colors[t_idx % len(colors)]
                        ls.style.linestyle.width = 3
                    if len(pts_boundary) >= 3:
                        hull = ConvexHull(np.array(pts_boundary))
                        hull_pts = np.vstack([np.array(pts_boundary)[hull.vertices], np.array(pts_boundary)[hull.vertices[0]]])
                        poly = fol.newpolygon(name=f"BOUNDARY_{t['name']}")
                        poly.outerboundaryis = [(p[0], p[1]) for p in hull_pts]
                        poly.style.polystyle.color = simplekml.Color.changealphaint(50, colors[t_idx % len(colors)])

                # Tambahkan Uncovered
                for j, rumah in gdf_rumah.iterrows():
                    if j not in taken_homes:
                        csv_data.append({'KODE_TIANG': 'TIDAK TERCOVER', 'NAMA_RUMAH': rumah.get('Name', f"H-{j+1}"), 'NAMA_JALAN': 'N/A', 'JARAK_KABEL_M': 0, 'LONGITUDE': rumah.geometry.x, 'LATITUDE': rumah.geometry.y})

                # Result
                df_csv = pd.DataFrame(csv_data)
                kml_path = "ISP_Optimized.kml"
                kml_out.save(kml_path)
                st.balloons()
                st.download_button("📥 Download KMZ", open(kml_path, "rb"), file_name="ISP_Final.kmz")
                st.download_button("📥 Download CSV", df_csv.to_csv(index=False).encode('utf-8'), "Data_ISP.csv", "text/csv")
                st.dataframe(df_csv)