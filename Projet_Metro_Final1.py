#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 14:29:20 2019

@author: elvinagovendasamy
"""



import csv
import pandas as pd
import matplotlib.pyplot as plt
import math
import networkx as nx
import seaborn as sns
import numpy as np
from collections import defaultdict
import mpu
from itertools import islice

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# using relative path

#path1 = '/Users/Nadia/Documents/Maths/PROJET_METRO_PYTHON/clear_data/base_FINAL.csv'
#path2 = '/Users/Nadia/Documents/Maths/PROJET_METRO_PYTHON/clear_data/ordre_stations_FINAL.csv'

path1='/Users/elvinagovendasamy/Projet_Metro/base_FINAL.csv'
path2='/Users/elvinagovendasamy/Projet_Metro/ordre_stations_FINAL.csv'

# Permet la lectures des fichiers metros et ordre_station et sort des dataframes. Enleve les doublons du fichier ordre_station.

class Reseau: # lecture des 2 fichiers
    def __init__(self):
        pass

    def reading_metros(self,filename1):
        self.metros = pd.read_csv(path1, sep = ";" )
        return self.metros

    def reading_ordres(self,filename2):
        self.ordre_station=pd.read_csv(path2, sep = ";" )
        #self.ordre_station.drop_duplicates(subset='Station',inplace=True)
        return self.ordre_station

    def station_names(self,filename2):
        self.ordre=Reseau.reading_ordres(self,filename2)
        self.names=list(self.ordre.Station)
        return self.names



class Graphique_Reseau:
    def __init__(self,filename1,filename2):
        self.ordre_station=Reseau.reading_ordres(self,filename2)
        self.metros=Reseau.reading_metros(self,filename1)
        self.names=Reseau.station_names(self,filename2)
        self.lignes=list(self.ordre_station['res_com'].unique())
        self.y1=list(self.ordre_station['latitude station'])
        self.x1=list(self.ordre_station['longitude station'])
        self.station=list(self.ordre_station.Station)
        self.station_suivant=list(self.ordre_station['Station suivante'])



        self.pos_all={}


    # Positionne tous les noeuds (stations) en utilisant les coordonnées
    def station_location(self):
        self.X=list(self.ordre_station['longitude station'])
        self.Y=list(self.ordre_station['latitude station'])
        for i in range(0, len(self.names)):
            self.pos_all[self.names[i]] = (self.X[i], self.Y[i])
        return self.pos_all


    # Permet de dessiner toutes les stations

    def drawing_correspondance(self):

        self.g=nx.Graph()
        cor=list(self.metros[pd.isna(self.metros['C_2']) == False].station)
        lat_cor=Shortest_Route.coordinates_liste_transfers(self)[1]
        lon_cor=Shortest_Route.coordinates_liste_transfers(self)[0]
        pos_cor={}
        for k in range(len(cor)):
            pos_cor[cor[k]]=(lat_cor[k],lon_cor[k])
        self.g.add_node(cor[0])
        self.g.add_nodes_from(cor[1:])
#        all_nodes=nx.draw_networkx_nodes(self.g, pos=Graphique_Reseau.station_location(self), node_size=5, node_color='k', alpha=0.5)
        correspondance1=nx.draw_networkx_nodes(self.g, pos=pos_cor, node_size=50, node_color='grey', alpha=0.25)
        correspondance2=nx.draw_networkx_labels(self.g,pos=pos_cor,font_size=4.0,font_color='grey')
#        plt.show()
        return correspondance1,correspondance2
    
    
    # Creation des aretes et coordonnées pour calculer les coordonnées de source et target
    def edgelist_walk(self,source_x,source_y,target_x,target_y):
        walk=list(Shortest_Route.shortest_route_weighted(self,source_x,source_y,target_x,target_y))
        n=len(walk)
        walk_edges=[]
        walk_edges.append((walk[0],walk[1]))
        walk_edges.append((walk[n-2],walk[n-1]))
        return walk_edges
    
    
    def drawing_first_stations(self):
        colors= ['#FFCD00','#C9910D','#704B1C','#007852','#6EC4E8','#62259D','#003CA6','#837902','#6EC4E8','#CF009E','#FF7E2E','#6ECA97','#FA9ABA','#6ECA97','#E19BDF','#B6BD00']
        lines=['M1', 'M10', 'M11', 'M12', 'M13', 'M14', 'M2', 'M3', 'M3 bis', 'M4', 'M5', 'M6', 'M7', 'M7 bis', 'M8', 'M9']
        first_=[]
#        reseau=[]
        lat=[]
        lon=[]
        for i in range(len(lines)):
            if self.ordre_station[self.ordre_station['res_com']==self.lignes[i]].res_com.values.any()==self.lignes[i]:
                m=list(self.ordre_station[self.ordre_station['res_com']==lines[i]].Station)
                y=list(self.ordre_station.loc[self.ordre_station['res_com']==self.lignes[i],'latitude station'])
                x=list(self.ordre_station.loc[self.ordre_station['res_com']==self.lignes[i],'longitude station'])
                first_.append(m[0])
                lon.append(y[0])
                lat.append(x[0])
#                reseau_list=list(self.ordre_station[self.ordre_station['res_com']==lines[i]].res_com)
#                reseau.append(reseau_list[0])
            pos_first_={}
            for k in range(len(first_)):
                pos_first_[first_[k]]=(lat[k],lon[k])
        nodeColorDict=dict(zip(first_,colors))
        self.g.add_node(first_[0])
        self.g.add_nodes_from(first_[1:])
        for nt in lines:
            ncolor=nodeColorDict[nt]
            first=nx.draw_networkx_nodes(self.g, pos=pos_first_, nodelist=first_,node_size=2, node_color=ncolor, label=nt,alpha=0.25)
        return first
    
#    def coordinates_start_end(self,source_x,source_y,target_x,target_y):
#        return {'start':(source_x,source_y),'end':(target_x,target_y)}

    
    def drawing_edges(self):
        #colors=['blue','skyblue','purple','pink','orange','magenta','green','lightgreen','darkgreen','red','yellow','grey','brown','darkblue','violet','aqua']
        colors= ['#FFCD00','#C9910D','#704B1C','#007852','#6EC4E8','#62259D','#003CA6','#837902','#6EC4E8','#CF009E','#FF7E2E','#6ECA97','#FA9ABA','#6ECA97','#E19BDF','#B6BD00']

        lines=['M1', 'M10', 'M11', 'M12', 'M13', 'M14', 'M2', 'M3', 'M3 bis', 'M4', 'M5', 'M6', 'M7', 'M7 bis', 'M8', 'M9']

        for i in range(len(self.lignes)):
            n=len(self.ordre_station[self.ordre_station['res_com']==self.lignes[i]].res_com)
            
            
            
            if self.ordre_station[self.ordre_station['res_com']==self.lignes[i]].res_com.values.any()==self.lignes[i]:
                y=list(self.ordre_station.loc[self.ordre_station['res_com']==self.lignes[i],'latitude station'])
                x=list(self.ordre_station.loc[self.ordre_station['res_com']==self.lignes[i],'longitude station'])
                m=list(self.ordre_station[self.ordre_station['res_com']==self.lignes[i]].Station)
                m_s=list(self.ordre_station.loc[self.ordre_station['res_com']==self.lignes[i],'Station suivante'])
                pos={}
                for k in range(n):
                    pos[m[k]]=(x[k],y[k])
                    self.g=nx.Graph()
                    self.g.add_node=m[k]

                for j in range(n):
                    self.g.add_edge(m[0],m_s[0])
                    self.g.add_edge(m[j],m_s[j])
                    nodes = nx.draw_networkx_nodes(self.g, pos=pos, node_size=7, node_color=colors[i])
                    edges= nx.draw_networkx_edges(self.g, pos=pos, edge_color=colors[i], width=0.5, alpha=0.25)
#                    nodes=nx.draw_networkx_nodes(self.g, pos=pos_line, node_size=10, node_color=colors[i],label=lines[i])
#            plt.legend(nodes,lines)
#         afficher la legende
                

        return nodes, edges


    def shortest_route_drawing(self,source_x,source_y,target_x,target_y):

        station_list = Shortest_Route.shortest_route_weighted(self,source_x,source_y,target_x,target_y)

        y1=list(self.metros['Latitude'])
        x1=list(self.metros['Longitude'])
        m=list(self.metros.station)
        # creating positions of shortest route
        lat=[]
        lon=[]
        for i in range(1,len(station_list)-1):
            for j in range(len(y1)):
                if station_list[i]==m[j]:
                    lat.append(x1[j])
                    lon.append(y1[j])
        lat=[source_x,*lat,target_x]
        lon=[source_y,*lon,target_y]

        pos_short={}
        for k in range(len(station_list)):
            pos_short[station_list[k]]=(lat[k],lon[k])
        
        edge_walk=Graphique_Reseau.edgelist_walk(self,source_x,source_y,target_x,target_y)
        walk=[i for i in edge_walk]
        walk=list(walk[0]+walk[1])
        print(walk)
        stations=[]
        stations.append(walk[1])
        stations.append(walk[2])
        print(stations)
        d_int=Graphique_Reseau.station_location(self)
        lat=[source_x,target_x]
        lon=[source_y,target_y]
        for items in stations:
            lat.append(d_int.get(items)[1])
            lon.append(d_int.get(items)[0])

        print(lat)
        print(lon)
        pos_walk_={}
        for k in range(len(walk)):
            pos_walk_[walk[k]]=(lat[k],lon[k])
        print(pos_walk_)
        

        
        self.g=nx.Graph()
        for i in range(len(station_list)):
            self.g.add_path(station_list)
#        self.h=nx.Graph()
#        for i in range(len(walk)):
#            self.h.add_path(walk)
        
        n2=nx.draw_networkx_nodes(self.g,pos=pos_short,node_size=25,node_color='blue')
        n3=nx.draw_networkx_labels(self.g,pos=pos_short,font_size=5,font_color='k')
        n4=nx.draw_networkx_edges(self.g,pos=pos_short,width=2.5,edge_color='black',style='dashed',alpha=1)
#        n5=nx.draw_networkx_nodes(self.g,pos=pos_walk_,node_size=35,node_color='red')
#        l=nx.draw_networkx_labels(self.g,pos=pos_walk_,font_size=10,font_color='red')
        
        return n2,n3,n4#,n5,l

#    def start_end_drawing(self,source_x,source_y,target_x,target_y):
#        edge_walk=Graphique_Reseau.edgelist_walk(self,source_x,source_y,target_x,target_y)
#        walk=[i for i in edge_walk]
#        walk=list(walk[0]+walk[1])
#        print(walk)
#        stations=[]
#        stations.append(walk[1])
#        stations.append(walk[2])
#        print(stations)
#        d_int=Graphique_Reseau.station_location(self)
#        lat=[source_x,target_x]
#        lon=[source_y,target_y]
#        for items in stations:
#            lat.append(d_int.get(items)[1])
#            lon.append(d_int.get(items)[0])
#
#        print(lat)
#        print(lon)
#        pos_walk_={}
#        for k in range(len(walk)):
#            pos_walk_[walk[k]]=(lat[k],lon[k])
#        print(pos_walk_)
#        
#        self.g=nx.Graph()
#        self.g.add_path(walk)
#        self.g.add_edge(walk[0],walk[1])
#        self.g.add_edge(walk[2],walk[3])
#        
#        n=nx.draw_networkx_nodes(self.g,pos=pos_walk_,node_size=35,node_shape='X',node_color='r')
#        l=nx.draw_networkx_labels(self.g,pos=pos_walk_,font_size=10,font_color='r')
#        e=nx.draw_networkx_edges(self.g,pos=pos_walk_,width=2.5,edge_color='r',style='dotted')
#        
#        return n,l,e
    
    
    def walking_route(self,source_x,source_y,target_x,target_y):
        extremes=['start','end']
        lat=[source_x,target_x]
        lon=[source_y,target_y]
        pos_walk={}
        for k in range(len(extremes)):
            pos_walk[extremes[k]]=(lat[k],lon[k])
        h=nx.Graph()
        for i in range(len(extremes)):
            h.add_path(extremes)
            nx.draw_networkx_nodes(h,pos=pos_walk,node_size=7,node_shape='s')
            nx.draw_networkx_edges(h,pos=pos_walk,width=1.0,edge_color='g',style='dotted')
            nx.draw_networkx_labels(h,pos=pos_walk,font_size=7,font_color='k')
#            plt.show()


    def combining_graphs(self,source_x,source_y,target_x,target_y):
        ds=min(Closest_Stations.distance_source_stations(self,source_x,source_y))
        dt=min(Closest_Stations.distance_target_stations(self,target_x,target_y))
        dist=Closest_Stations.distance_start_end(self,source_x,source_y,target_x,target_y)
        if dist<ds+dt:
            return Graphique_Reseau.drawing_correspondance(self),Graphique_Reseau.drawing_edges(self),Graphique_Reseau.walking_route(self,source_x,source_y,target_x,target_y)
        else:
            return Graphique_Reseau.drawing_correspondance(self),Graphique_Reseau.drawing_edges(self),Graphique_Reseau.shortest_route_drawing(self,source_x,source_y,target_x,target_y)#,Graphique_Reseau.start_end_drawing(self,source_x,source_y,target_x,target_y)




class Shortest_Route(Graphique_Reseau):

    def __init__(self,filename1,filename2):
        Graphique_Reseau.__init__(self,filename1,filename2)

#        self.ordre_station=Reseau.reading_ordres(self,filename2)
#        self.metros=Reseau.reading_metros(self,filename1)
#        self.station=list(self.ordre_station.Station)
#        self.station_suivant=list(self.ordre_station['Station suivante'])
#        self.y1=list(self.ordre_station['latitude station'])
#        self.x1=list(self.ordre_station['longitude station'])
#        self.names=Reseau.station_names(self,filename2)
#        self.pos_all={}




    def generating_initial_edge_list(self):
        edgelist=[]
        for i in range(len(self.station)):
            edgelist.append((self.station[i],self.station_suivant[i]))
        return edgelist


    def source_target_stations_path(self,source_x,source_y,target_x,target_y):
        new_edgelist=Closest_Stations.generating_new_edgelist(self,source_x,source_y,target_x,target_y)
        g=nx.Graph(new_edgelist)
        return (list(nx.shortest_path(g, source='start', target='end')))


    def shortest_route_weighted(self,source_x,source_y,target_x,target_y):

        transfers = Shortest_Route.liste_transfers(self)
        edges_sc = Shortest_Route.edgelist_NOtransfers(self)
        edge_c = Shortest_Route.edgelist_transfers(self)
        stations_source = Closest_Stations.stations_source(self,source_x,source_y)
        stations_target = Closest_Stations.stations_target(self,target_x,target_y)

        walk=['start','end']
        distances_source = Closest_Stations.distance_source_stations(self,source_x,source_y)
        distances_target = Closest_Stations.distance_target_stations(self,target_x,target_y)

        # Sélectionner la station la plus proche en se basant sur la distance
        index_source = np.argmin(distances_source)
        closest_source = stations_source[index_source]

        index_target = np.argmin(distances_target)
        closest_target = stations_target[index_target]

        g = nx.Graph(edges_sc)
        # Arete avec poids lié à la distance entre le début de la marche et la station la proche
        g.add_edge(walk[0],closest_source,weight=1/min(distances_source))
        # Arete avec poids lié à la distance entre la station d'arrivé et le point d'arrivé
        g.add_edge(closest_target,walk[1],weight=1/min(distances_target))

        metro_edgelist=Shortest_Route.generating_initial_edge_list(self)

        # liste de toutes les aretes
        g.add_path(metro_edgelist)

        # Trouver les arêtes avec correspondances pour ajuster le poids
        # On commence par sortir les 3 premiers chemins les plus courts
        for path in Shortest_Route.k_shortest_paths(self,g, 'start', 'end', 5):
            p=path

        transfer_list=[]
        avant=[]
        suivant=[]
        for j in range(len(p)-1):
            for i in range(len(transfers)):
                if p[j]==transfers[i]:
                    transfer_list.append(p[j])
                    avant.append(p[j-1])
                    suivant.append(p[j+1])

        c_edge=[]
        for i in range(len(suivant)):
            ligne_station_suivante = list(self.ordre_station.loc[self.ordre_station['Station suivante'] == suivant[i],'res_com'])
            transfer_linked=list(self.ordre_station.loc[self.ordre_station['Station suivante'] == suivant[i],'Station'])
            ligne_station_avant=list(self.ordre_station.loc[self.ordre_station['Station'] == avant[i],'res_com'])
            station_suivant=list(self.ordre_station.loc[self.ordre_station['Station suivante'] == suivant[i],'Station suivante'])

            for k in range(len(ligne_station_suivante)):
                if ligne_station_suivante[k]!=ligne_station_avant[k]:
                        c_edge.append((transfer_linked[k],station_suivant[k]))
        # ajout des edges avec correspondance, s'il n'y a pas de transfer, le poids reste par défaut à 1, lorsqu'il y a correspondances, le poids est mis à 3.
        g.add_edges_from(edge_c)
        g.add_edges_from(c_edge,weight=3)

        new_path=nx.shortest_path(g, source='start',target='end',weight='weight')
        return new_path

    

    def k_shortest_paths(self,G, source, target, k, weight=None):
        return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))


    def liste_transfers(self):
        return  list(self.metros[pd.isna(self.metros['C_2']) == False].station)  #stations avec correspondances

    def coordinates_liste_transfers(self):
        corr=Shortest_Route.liste_transfers(self)
        d_int=Graphique_Reseau.station_location(self)
        lat=[]
        lon=[]
        for items in corr:
            lat.append(d_int.get(items)[1])
            lon.append(d_int.get(items)[0])
        return lat,lon


    def liste_NOtransfers(self):
        return list(self.metros[pd.isna(self.metros['C_2']) == True].station) #stations sans correspondance
    


    def edgelist_transfers(self):
        liste_correspondance2=Shortest_Route.liste_transfers(self)
        lignes_correspondance=list(tuple(self.ordre_station.loc[self.ordre_station['Station'] == liste_correspondance2[i],'res_com'])  for i in range(len(liste_correspondance2)))
        edges_c = []
        liste_station_suivante = list(tuple(self.ordre_station.loc[self.ordre_station['Station'] == liste_correspondance2[i],'Station suivante'])  for i in range(len(liste_correspondance2)))
        lignes_suivante=list(tuple(self.ordre_station.loc[self.ordre_station['Station suivante'] == liste_station_suivante[i],'res_com'])  for i in range(len(liste_station_suivante)))
        for i in range(len(liste_correspondance2)):
            for k in range(len(liste_station_suivante[i])):
                if lignes_correspondance[i]!=lignes_suivante[i]: # on ajoute la condition pour prendre en compte les changements de lignes
                    edges_c.append((liste_correspondance2[i],liste_station_suivante[i][k]))
        edges_c1 = []
        for i in range(len(liste_correspondance2)):
            for k in range(len(liste_station_suivante[i])):
                    edges_c1.append((liste_correspondance2[i],liste_station_suivante[i][k]))
        return edges_c


    def edgelist_NOtransfers(self):
        liste_sans_correspondance2 = Shortest_Route.liste_NOtransfers(self)
        liste_station_suivante_sc = list(tuple(self.ordre_station.loc[self.ordre_station['Station'] == liste_sans_correspondance2[i],'Station suivante']) for i in range (len(liste_sans_correspondance2)))
        edges_sc = []
        for i in range(len(liste_sans_correspondance2)):
            for k in range(len(liste_station_suivante_sc[i])):
                edges_sc.append((liste_sans_correspondance2[i],liste_station_suivante_sc[i][k]))
        return edges_sc

    def edgelist_source_target(self,source_x,source_y,target_x,target_y):
        source=list(set(Closest_Stations.stations_source(self,source_x,source_y)))
        target=list(set(Closest_Stations.stations_target(self,target_x,target_y)))
        new_edges=[]
        extremes=['start','end']

        for i in range(len(source)):
            new_edges.append([extremes[0],source[i]])
        for j in range(len(target)):
            new_edges.append([extremes[1],target[j]])
        return new_edges
    
    




class Closest_Stations(Shortest_Route):
    def __init__(self,filename1,filename2):
        super().__init__(filename1,filename2)


#        self.ordre_station=Reseau.reading_ordres(self,filename2)
#        self.station=list(self.ordre_station.Station)
#        self.station_suivant=list(self.ordre_station['Station suivante'])
#        self.y1=list(self.ordre_station['latitude station'])
#        self.x1=list(self.ordre_station['longitude station'])
#        self.names=Reseau.station_names(self,filename2)
#        self.pos_all={}


    def stations_source(self,source_x,source_y):
        closest_stations_source=[]
        for j in range(len(self.y1)):
            dist_y=(source_y-self.y1[j])**2
            dist_x=(source_x-self.x1[j])**2
            dist=np.sqrt(dist_y+dist_x)
            if dist<0.045:
                closest_stations_source.append(self.station[j])
        return closest_stations_source



    def stations_target(self,target_x,target_y):
        closest_stations_target=[]

        for j in range(len(self.y1)):
            dist_y=(target_y-self.y1[j])**2
            dist_x=(target_x-self.x1[j])**2
            dist=np.sqrt(dist_y+dist_x)
            if dist<0.045:
                closest_stations_target.append(self.station[j])
        return closest_stations_target


    def coordinates_closest_station(self,source_x,source_y,target_x,target_y):
        T=Closest_Stations.stations_target(self,target_x,target_y)
        S=Closest_Stations.stations_source(self,source_x,source_y)
        all_closest=T+S
        d_int=Graphique_Reseau.station_location(self)
        lat=[]
        lon=[]
        for items in all_closest:
            lat.append(d_int.get(items)[0])
            lon.append(d_int.get(items)[1])
        return lat,lon


    def coordinates_closest_station_source(self,source_x,source_y):
        S=Closest_Stations.stations_source(self,source_x,source_y)
        d_int=Graphique_Reseau.station_location(self)
        lat=[]
        lon=[]
        for items in S:
            lat.append(d_int.get(items)[0])
            lon.append(d_int.get(items)[1])
        return lat,lon


    def coordinates_closest_station_target(self,target_x,target_y):
        T=Closest_Stations.stations_target(self,target_x,target_y)
        d_int=Graphique_Reseau.station_location(self)
        lat=[]
        lon=[]
        for items in T:
            lat.append(d_int.get(items)[0])
            lon.append(d_int.get(items)[1])
        return lat,lon



    def generating_new_edgelist(self,source_x,source_y,target_x,target_y):
        source=list(set(Closest_Stations.stations_source(self,source_x,source_y)))
        target=list(set(Closest_Stations.stations_target(self,target_x,target_y)))
        new_edges=Shortest_Route.generating_initial_edge_list(self)
        extremes=['start','end']
        for i in range(len(source)):
            new_edges.append((extremes[0],source[i]))
        for j in range(len(target)):
            new_edges.append((extremes[1],target[j]))
        return new_edges



    def distance_target_stations(self,target_x,target_y):
        dist_target=[]
        lat=Closest_Stations.coordinates_closest_station_target(self,target_x,target_y)[0]
        lon=Closest_Stations.coordinates_closest_station_target(self,target_x,target_y)[1]
        for j in range(len(lat)):
            dist_target.append(mpu.haversine_distance((target_x,target_y),(lat[j],lon[j])))
        return dist_target



    def distance_source_stations(self,source_x,source_y):
        lat=Closest_Stations.coordinates_closest_station_source(self,source_x,source_y)[0]
        lon=Closest_Stations.coordinates_closest_station_source(self,source_x,source_y)[1]
        dist_source=[]
        for j in range(len(lat)):
            dist_source.append(mpu.haversine_distance((source_x,source_y),(lat[j],lon[j])))
        return dist_source


    def distance_start_end(self,source_x,source_y,target_x,target_y):
        return mpu.haversine_distance((target_x,target_y),(source_x,source_y))



    def temps_en_secondes(self, source_x,source_y,target_x,target_y):
        distance = Closest_Stations.distance_entre_points(self, source_x, source_y, target_x, target_y)
        vitesse_marcheur = 1.67   #en mètres par seconde
        temps = distance / vitesse_marcheur   #en secondes
        return temps   #en secondes

    def distance_entre_points(self,source_x, source_y, target_x,target_y):
        latitude_point1 = source_x
        longitude_point1 = source_y
        latitude_point2 = target_x
        longitude_point2 = target_y
        R = 6372800  #en mètres
        phi1, phi2 = math.radians(latitude_point1), math.radians(latitude_point2)
        dphi  = math.radians(latitude_point2 - latitude_point1)
        dlambda    = math.radians(longitude_point1 - longitude_point2)
        a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
        distance = 2*R*math.atan2(math.sqrt(a), math.sqrt(1 - a))  #en mètres
        return distance


    def temps_total(self,source_x,source_y,target_x,target_y): #recuperer la liste des stations qui passent:
        temps_entre_stations = 60 #secondes
        temps_correspondance = 160
        stations_visitees =  Shortest_Route.shortest_route_weighted(self,source_x,source_y,target_x,target_y)
        temps_total = 0
        if len(stations_visitees) == 2:  #il n'y a que start et end donc pas de stations visitées
            temps_de_marche = Closest_Stations.temps_en_secondes(self,source_x,source_y,target_x,target_y)
            temps_total = temps_de_marche // 60
            print("Temps de marche estimé: ", temps_total, "minutes")
        else:
            S_source = stations_visitees[1]  #station_source la plus proche du point de depart
            S_target = stations_visitees[-2]  #station_target la plus proche du point d'arrivée
            d_int = Graphique_Reseau.station_location(self)
            lat_source = d_int.get(S_source)[0]  #latitude station source
            lon_source = d_int.get(S_source)[1]  #longitude station source
            lat_target = d_int.get(S_target)[0]   #latitude station target
            lon_target = d_int.get(S_target)[1]   # longitude station target
            temps_de_marche_source = Closest_Stations.temps_en_secondes(self, source_x, source_y,lat_source, lon_source)
            print("Rejoindre la station", S_source, "\n Temps de marche estimé: ", temps_de_marche_source//60, "minutes", )
            temps_de_marche_target = Closest_Stations.temps_en_secondes(self, target_x,target_y, lat_target, lon_target)
            temps_total = temps_de_marche_source + temps_de_marche_target

            for i in range (2,len(stations_visitees)-2):
                stationA = stations_visitees[i-1]
                stationB = stations_visitees[i+1]
                if str(self.metros.loc[self.metros['station'] == stationA].res_com) == str(self.metros.loc[self.metros['station'] == stationB].res_com):   #meme res_com alors ajouter temps entre stations sinon
                    temps_total += temps_entre_stations
                elif str(self.metros.loc[self.metros['station'] == stationA].res_com) != str(self.metros.loc[self.metros['station']== stationB].res_com):
                    temps_total += (temps_entre_stations + temps_correspondance/2)
            temps_total_minutes = temps_total / 60
            temps_total_secondes = temps_total
            print('Le trajet total a une durée de: ', temps_total_minutes,' minutes')
            return(temps_total_secondes)




# CALCUL DU TEMPS: A COMPLETER POUR L'AFFICHAGE A LA FIN

#    def temps_en_secondes(self, source_x,source_y,target_x,target_y):
#        distance = Shortest_Route.distance_entre_points2(self, source_x, source_y, target_x, target_y)
#        vitesse_marcheur = 1,67   #en mètres par seconde
#        temps = distance / vitesse_marcheur   #en secondes
#        return temps
#
#    def temps_en_secondes2(self, distance):
#        vitesse_marcheur = 1.67
#        temps = distance/vitesse_marcheur
#        return temps
#
#    def temps_en_minutes(self,source_x,source_y,target_x,target_y):
#        temps_en_secondes = Shortest_Route.temps_en_secondes(self,source_x,source_y,target_x,target_y)
#        temps = temps_en_secondes / 60
#        return temps




if __name__=='__main__':

    stat=Closest_Stations(path1,path2)
    """ Il faudra demander les coordonnees a 3d.p. dans le terminal"""
    coordinates=[2.2,48.75,2.201,48.754]

    dessin=Graphique_Reseau(path1,path2)
    short=Shortest_Route(path1,path2)

#    transfers=dessin.drawing_correspondance()

    edge_walk=short.edgelist_walk(coordinates[0],coordinates[1],coordinates[2],coordinates[3])
#    path_final=short.source_target_stations_path(coordinates[0],coordinates[1],coordinates[2],coordinates[3])
#    corres=short.shortest_route_weighted(coordinates[0],coordinates[1],coordinates[2],coordinates[3])
    reseau=dessin.drawing_edges()
#    draw=dessin.start_end_drawing(coordinates[0],coordinates[1],coordinates[2],coordinates[3])
    #graph=dessin.combining_graphs(coordinates[0],coordinates[1],coordinates[2],coordinates[3])
#    test = stat.temps_total(coordinates[0],coordinates[1],coordinates[2],coordinates[3])
    try:
        graph=dessin.combining_graphs(coordinates[0],coordinates[1],coordinates[2],coordinates[3])
    except ValueError as e:
        print("Il n'y a pas de station de métro à proximité, veuillez entrer d'autres coordonnées ")
    plt.show()
