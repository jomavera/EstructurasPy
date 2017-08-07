# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 19:24:29 2017

@author: Jose Manuel
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

class Marco2D():

    def __init__(self,C=np.empty(shape=(1,3)),
                 E=np.empty(shape=(1,5)), R=np.empty(shape=(1,4)),
                    F=np.empty(shape=(1,4)),Q=np.empty(shape=(1,1)),corte=0):
        """ Argumentos
            -------------------------------------------------------
            C: Matriz de coordenadas (num_elmentos, 2) = [x,y]
            E: Matriz de propiedades de elementos (num_elementos,6)= [nodo inicial,
                nodo final,area de la seccion, inercia flexural,
                modulo de elasticidad, modulo de corte]
            R: Matriz de resticiones. Indica nodos vinculados si esta restringido es 1, 0 si esta libre
                (nodos,4)=[nodo vinculado, dx,dy,tz]
            F: Matriz de fuerzas externas(nodos cargados,4)=[nodo, Fjx,Fjy,Mjz]
                Si dim=2 Fjz=0
            Q: Vector de cargas distribuidas de vano aplicadas perpendicularmente
                sobre las barras (num_elementos, 1)= [qj]
            corte: Indicador si incluye deformaciones por corte
                    0: No se incluye
                    1:Si se incluye
            """

        self.C=C
        self.E=E
        self.R=R
        self.F=F
        self.Q=Q
        self.corte=corte
        self.num_nodos=self.C.shape[0]
        self.num_elem=self.E.shape[0]
        self.k=1.2

    def cargar_csv(self,directorio=None):
        if directorio==None:
            print('Error, indique directorio donde se encuentran los archivos de datos')
        else:
            files=['C.csv','E.csv','R.csv','F.csv','Q.csv']
            for file in files:
                filename=os.path.join(directorio, file)
                if file=='C.csv':
                    temp=np.genfromtxt(filename, delimiter=';')
                    np.resize(self.C,(temp.shape[0],temp.shape[1]))
                    self.C=temp
                    self.num_nodos=self.C.shape[0]
                elif file=='E.csv':
                    temp=np.genfromtxt(filename, delimiter=';',dtype=np.float_)
                    np.resize(self.E,(temp.shape[0],temp.shape[1]))
                    self.E=temp
                    self.num_elem=self.E.shape[0]
                elif file=='R.csv':
                    temp=np.genfromtxt(filename, delimiter=';',dtype=np.float_)
                    np.resize(self.R,(temp.shape[0],temp.shape[1]))
                    self.R=temp
                elif file=='F.csv':
                    temp=np.genfromtxt(filename, delimiter=';')
                    try:
                        np.resize(self.F,(temp.shape[0],temp.shape[1]))
                        self.F=temp
                        self.num_fuerzas=self.F.shape[0]
                    except:
                        np.resize(self.F,(1,temp.shape[0]))
                        self.F=temp.reshape((1,self.F.shape[1]))
                        self.num_fuerzas=1
                elif file=='Q.csv':
                    temp=np.genfromtxt(filename, delimiter=';',dtype=np.float_)
                    np.resize(self.Q,(temp.shape[0],))
                    self.Q=temp

        self.num_nodos=self.C.shape[0]
        self.num_elem=self.E.shape[0]


    def analizar(self):
        """ Retorna
            -------------------------------------------------------
            S=Matriz de esfuerzos en elementos [SF, ST] ; SF=Por fuerzas externas,
                ST=Por temperatura (elementos,2)
            R=Matriz de reacciones [RF,RT]; RF= reacciones por fuerzas externas,
                RT=Reacciones por temperatura (nodos vinculados,7)
            u=Matriz de desplazamientos nodales [uF,uT](numero nodos,6)
            """

        self.Kext=np.zeros((3*self.num_nodos,3*self.num_nodos))
        self.Next=np.zeros((3*self.num_nodos,1))


        #Se determina matriz de rigidez global
        elementos_Dict={}
        for elem in range(self.E.shape[0]):

            barra=self.E[elem,:]
            elementos_Dict[elem]={}

            ni=(barra[0].astype(int)-1)
            nj=(barra[1].astype(int)-1)
            ri=(self.C[ni,:]).T.astype(np.float_)   #coordenadas iniciales del nodo
            rj=(self.C[nj,:]).T.astype(np.float_)   #coordenadas finales del nodo

            L=np.linalg.norm(ri-rj).astype(np.float_)    #longitud del nodo
            d=(ri-rj)/L
            A=barra[2].astype(np.float_)
            I=barra[3].astype(np.float_)
            E=barra[4].astype(np.float_)
            G=barra[5].astype(np.float_)

            q=self.Q[elem]

            phi=self.k*(12*E*I/(G*A*L**2))*self.corte;
            klocal=np.array([[E*A/L, 0, 0, -E*A/L, 0, 0],
                    [0, 12*E*I/((1+phi)*L**3), 6*E*I/((1+phi)*L**2), 0, -12*E*I/((1+phi)*L**3), 6*E*I/((1+phi)*L**2)],
                    [0, 6*E*I/((1+phi)*L**2), (4+phi)*E*I/((1+phi)*L), 0, -6*E*I/((1+phi)*L**2), (2-phi)*E*I/((1+phi)*L)],
                    [-E*A/L, 0, 0, E*A/L, 0, 0],
                    [0, -12*E*I/((1+phi)*L**3), -6*E*I/((1+phi)*L**2), 0, 12*E*I/((1+phi)*L**3), -6*E*I/((1+phi)*L**2)],
                    [0, 6*E*I/((1+phi)*L**2), (2-phi)*E*I/((1+phi)*L), 0, -6*E*I/((1+phi)*L**2), (4+phi)*E*I/((1+phi)*L)]])


            nlocal=np.array([0, 12*q*L, 12*q*L**2, 0, 12*q*L, -12*q*L**2]).T

            Transf=np.array([[d[0],d[1],0,0,0,0],
                         [-d[1],d[0],0,0,0,0],
                         [0, 0, 1, 0, 0, 0],
                         [0, 0, 0, d[0],d[1], 0],
                         [0, 0, 0, -d[1], d[0], 0],
                         [0,0,0,0,0, 1]])

            Ext=np.zeros((6,3*self.num_nodos))
            Ext[:3,(3*(ni+1)-3):(3*(ni+1))]=np.identity(3)
            Ext[3:,(3*(nj+1)-3):(3*(nj+1))]=np.identity(3)

            kglobal=np.dot(Transf.T,np.dot(klocal,Transf))
            Nglobal=np.dot(Transf.T,nlocal)

            elementos_Dict[elem]['A']=A
            elementos_Dict[elem]['E']=E
            elementos_Dict[elem]['L']=L
            elementos_Dict[elem]['I']=I
            elementos_Dict[elem]['G']=G
            elementos_Dict[elem]['T']=Transf
            elementos_Dict[elem]['klocal']=klocal
            elementos_Dict[elem]['nlocal']=nlocal
            elementos_Dict[elem]['Ext']=Ext


            part_1=np.dot(Transf,Ext) #6x6 x 6x30 =6x30
            part_2=np.dot(kglobal,part_1) #6x6 x 6x30 = 6x30
            part_3=np.dot(Transf.T,part_2) #6x30

            self.Kext=self.Kext+np.dot(Ext.T,part_3) #6x6
            part_4=np.dot(Ext.T,np.dot(Transf.T,Nglobal)).reshape((30,1))
            self.Next=np.add(self.Next,part_4)



        #Se crean las sub-matrices K_ll, K_lv, K_vv
        nodos_restringidos=[]
        nodos_libres=[]
        lista_nodosrestringidos=self.R[:,0].astype(int)
        row=0

        for nodo in range(self.num_nodos):
            if nodo+1 in lista_nodosrestringidos:
                for dim in range(1,4):
                    if self.R[row,dim] == 1:
                        nodos_restringidos.append((nodo)*3+dim-1)
                    else:
                        nodos_libres.append((nodo)*3+dim-1)
                row +=1
            else:
                for dim in range(1,4):
                    nodos_libres.append(nodo*3+dim-1)

        Kll=self.Kext[np.ix_(nodos_libres,nodos_libres)]
        Klv=self.Kext[np.ix_(nodos_libres,nodos_restringidos)]
        Kvl=self.Kext[np.ix_(nodos_restringidos,nodos_libres)]
        Kvv=self.Kext[np.ix_(nodos_restringidos,nodos_restringidos)]

        Nl=self.Next[np.ix_(nodos_libres)]
        Nv=self.Next[np.ix_(nodos_restringidos)]

        reacciones=np.zeros((3*self.num_nodos,1))

        for f in range(self.F.shape[0]):
            nc=self.F[f,0].astype(int)-1
            reacciones[(3*(nc+1)-3):(3*(nc+1)),:]=self.F[f,1:].reshape((3,1))

        Fe=reacciones[np.ix_(nodos_libres)]+Nl

        u_F=np.linalg.solve(Kll,Fe)
        u_N=np.linalg.solve(-Kll,Fe)

        reacc_F=np.dot(Kvl,u_F)
        reacc_N=Nv+np.dot(Kvl,u_N)

        self.U=np.zeros((self.num_nodos,3))

        for nodo in range(len(nodos_libres)):
            n_in=nodos_libres[nodo]
            row=np.floor((n_in)/3).astype(int)
            col=n_in-3*row
            self.U[row,col]=u_F[nodo]+u_N[nodo]

        self.Reac=np.zeros((self.R.shape[0],4))
        actual_reac = 0
        for row in range( self.R.shape[0]):
            r = np.zeros((1,3))
            for column in range(1,4):
                if self.R[row, column]== 1:
                    r[0,column - 1] = reacc_N[actual_reac,0]
                    actual_reac = actual_reac + 1
            union=np.insert(r,0,self.R[row,0])

            self.Reac[row, :] = union

        r=np.zeros((3*self.num_nodos,1))
        r[np.ix_(nodos_libres)]=u_F


        self.S=np.zeros((self.num_elem,6))
        for elem in range(self.E.shape[0]):
            A=elementos_Dict[elem]['A']
            E=elementos_Dict[elem]['E']
            L=elementos_Dict[elem]['L']
            I=elementos_Dict[elem]['I']
            G=elementos_Dict[elem]['G']
            Transf=elementos_Dict[elem]['T']
            klocal=elementos_Dict[elem]['klocal']
            nlocal=elementos_Dict[elem]['nlocal']
            Ext=elementos_Dict[elem]['Ext']

            v_F=np.dot(Transf,np.dot(Ext,r))    #desplazamientos axiales de los nodos de la barra en ejes locales


            self.S[elem,:]=np.add(np.dot(klocal,v_F).T,nlocal)

        return self.S,self.Reac,self.U

