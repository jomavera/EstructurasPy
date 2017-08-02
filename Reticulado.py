# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 16:28:44 2017

@author: Jose Manuel
"""
import numpy as np
import pandas as pd
import sympy
from scipy.linalg import lu
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import os

class Reticulado():

    def __init__(self,C=np.empty(shape=(1,3)),
                 E=np.empty(shape=(1,5)), R=np.empty(shape=(1,4)),
                    F=np.empty(shape=(1,4)),dT=np.empty(shape=(1,1)),dim=2, graf=False):
        """ Argumentos
            -------------------------------------------------------
            C: Matriz de coordenadas (num_elmentos, 3) = [x,y,z]
            E: Matriz de propiedades de elementos (num_elementos,5)= [nodo inicial, nodo final,
                area de la seccion, modulo de elasticidad, coef. dilatacion]
            R: Matriz de resticiones. Indica nodos vinculados si esta restringido es 1, 0 si esta libre
                (nodos,4)=[nodo vinculado, dx,dy,dz]
            F: Matriz de fuerzas externas(nodos cargados,4)=[nodo, Fjx,Fjy,Fjz] Si dim=2 Fjz=0
            dT: Vector de cambios de temperatura en elementos (num_elementos, 1)= [dTj]
            dim: Indicador de reticulado 2D o 3D
            graf:0 no se grafica, 1 si se grafica

            """

        self.C=C
        self.E=E
        self.R=R
        self.F=F
        self.dT=dT
        self.dim=dim
        self.graf=graf
        self.num_nodos=self.C.shape[0]
        self.num_elem=self.E.shape[0]

    def cargar_csv(self,directorio=None):
        if directorio==None:
            print('Error, indique directorio donde se encuentran los archivos de datos')
        else:
            files=['C.csv','E.csv','R.csv','F.csv','dT.csv']
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
                    np.resize(self.F,(temp.shape[0],temp.shape[1]))
                    self.F=temp
                elif file=='dT.csv':
                    temp=np.genfromtxt(filename, delimiter=';',dtype=np.float_)
                    np.resize(self.dT,(temp.shape[0],))
                    self.dT=temp


    def get_summary(self):
        df_C=pd.DataFrame(data=self.C,
                          index=np.linspace(1,self.C.shape[0],self.C.shape[0]).astype(int),
                                            columns=['x','y','z'])
        df_E=pd.DataFrame(data=self.E,
                          index=np.linspace(1,self.E.shape[0],self.E.shape[0]).astype(int),
                          columns=['Nodo Inicial','Nodo Final','Area','Elasticidad','Coef. Dilatacion'])

        df_R=pd.DataFrame(data=self.R,
                          index=np.linspace(1,self.R.shape[0],self.R.shape[0]).astype(int),
                          columns=['Nodo','dx','dy','dz'])

        df_F=pd.DataFrame(data=self.F,
                          index=np.linspace(1,self.F.shape[0],self.F.shape[0]).astype(int),
                          columns=['Nodo','Fjx','Fjy','Fjz'])

        df_dt=pd.DataFrame(data=self.dT,
                          index=np.linspace(1,self.dT.shape[0],self.dT.shape[0]).astype(int),
                          columns=['dT'])

    def rigidez(self):
        """ Retorna
            -------------------------------------------------------
            S=Matriz de esfuerzos en elementos [SF, ST] ; SF=Por fuerzas externas,
                ST=Por temperatura (elementos,2)
            R=Matriz de reacciones [RF,RT]; RF= reacciones por fuerzas externas,
                RT=Reacciones por temperatura (nodos vinculados,7)
            u=Matriz de desplazamientos nodales [uF,uT](numero nodos,6)
            """
        if self.dim==2:
            self.Kext=np.zeros((2*self.num_nodos,2*self.num_nodos))
            self.Next=np.zeros((2*self.num_nodos,1))
        else:
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
            d=(ri-rj)/L                #vector direccion
            if self.dim==2:
                d=d[:-1]
            A=barra[2].astype(np.float_)
            E=barra[3].astype(np.float_)

            #Matriz de rigidez local
            klocal=A*E/L*np.array([[1,-1],[-1,1]],dtype=np.float_)
            cof_dila=barra[4]
            nlocal=A*E*self.dT[elem]*cof_dila*np.array([[1],[-1]],dtype=np.float_)
            if self.dim==2:
                zeros_a=[0,0]
            else:
                zeros_a=[0,0,0]
            array_1=np.concatenate((d.T,zeros_a),axis=0)
            array_2=np.concatenate((zeros_a,d.T),axis=0)
            Transf=np.array([array_1,array_2])

            #Matriz de rigidez global
            kglobal=np.dot(Transf.T,np.dot(klocal,Transf))
            Nglobal=np.dot(Transf.T,nlocal)
            if self.dim==2:
                Ext=np.zeros((4,2*self.num_nodos))
                Ext[:2,(2*(ni+1)-2):(2*(ni+1))]=np.identity(2)
                Ext[2:,(2*(nj+1)-2):(2*(nj+1))]=np.identity(2)
            else:
                Ext=np.zeros((6,3*self.num_nodos))
                Ext[:3,(3*(ni+1)-3):(3*(ni+1))]=np.identity(3)
                Ext[3:,(3*(nj+1)-3):(3*(nj+1))]=np.identity(3)

            #almacenamos en un dictinario
            elementos_Dict[elem]['A']=A
            elementos_Dict[elem]['E']=E
            elementos_Dict[elem]['L']=L
            elementos_Dict[elem]['cof_dila']=cof_dila
            elementos_Dict[elem]['T']=Transf
            elementos_Dict[elem]['nlocal']=nlocal
            elementos_Dict[elem]['Ext']=Ext
            self.Kext=self.Kext+np.dot(Ext.T,np.dot(kglobal,Ext))
            self.Next=np.add(self.Next,np.dot(Ext.T,Nglobal))



        #Se crean las sub-matrices K_ll, K_lv, K_vv
        nodos_restringidos=[]
        nodos_libres=[]
        lista_nodosrestringidos=self.R[:,0].astype(int)
        row=0
        if self.dim==2:
            for nodo in range(self.num_nodos):
                if nodo+1 in lista_nodosrestringidos:
                    for dim in range(1,3):
                        if self.R[row,dim] == 1:
                            nodos_restringidos.append((nodo)*2+dim-1)
                        else:
                            nodos_libres.append((nodo)*2+dim-1)
                    row +=1
                else:
                    for dim in range(1,3):
                        nodos_libres.append(nodo*2+dim-1)

        else:

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


        #Se construye vector de reacciones
        reacciones=np.zeros((3*self.num_nodos,1))

        for f in range(self.F.shape[0]):
            nc=self.F[f,0].astype(int)-1
            reacciones[(3*(nc+1)-3):(3*(nc+1)),:]=self.F[f,1:].reshape((3,1))

        Fe=reacciones[np.ix_(nodos_libres)]

        #print(linalg.det(Kll))
        u_F=np.linalg.lstsq(Kll,Fe)[0]
        u_N=np.linalg.lstsq(-Kll,Nl)[0]
        #u_F=linalg.inv(Kll).dot(Fe)
        #u_F=sympy.Matrix(np.concatenate((Kll,Fe),axis=1)).rref()
        #u_N=sympy.Matrix(np.concatenate((-Kll,Nl),axis=1)).rref()


#        pl_1,u_F_exp=lu(np.concatenate((Kll,Fe),axis=1),permute_l=True)
#        u_F=u_F_exp[:,:Kll.shape[1]]
#        pl_2,u_N_exp=lu(np.concatenate((-Kll,Nl),axis=1),permute_l=True)
#        u_N=u_N_exp[:,:Kll.shape[1]]

        reacc_F=np.dot(Kvl,u_F)
        reacc_N=Nv+np.dot(Kvl,u_N)

        if self.dim ==2:
            U_F=np.zeros((self.num_nodos,2))
            U_N=np.zeros((self.num_nodos,2))
            #Insercion de valores de nodos libres en posiciones  de martiz global de desplazamientos
            for nodo in range(len(nodos_libres)):
                n_in=nodos_libres[nodo]
                row=np.floor((n_in)/2).astype(int)
                col=n_in-2*row
                U_F[row,col]=u_F[nodo]
                U_N[row,col]=u_N[nodo]
        else:
            U_F=np.zeros((self.num_nodos,3))
            U_N=np.zeros((self.num_nodos,3))

            #Insercion de valores de nodos libres en posiciones  de martiz global de desplazamientos
            for nodo in range(len(nodos_libres)):
                n_in=nodos_libres[nodo]
                row=np.floor((n_in)/3).astype(int)
                col=n_in-3*row
                U_F[row,col]=u_F[nodo]
                U_N[row,col]=u_N[nodo]

        self.U=np.zeros((self.num_nodos,6))

        if self.dim==2:
            self.U[:,:2]=U_F
            self.U[:,3:-1]=U_N
        else:
            self.U[:,:3]=U_F
            self.U[:,3:]=U_N

        #Transformacion de matriz U_F a vector
        vector_U_F=U_F.flatten().T
        vector_U_N=U_F.flatten().T

        #Transformacion de matriz R a vector
        R_F=np.zeros((self.R.shape[0],4))
        R_N=np.zeros((self.R.shape[0],4))
        pos=0
        for row in range(self.R.shape[0]):
            r_F=np.zeros((1,3))
            r_N=np.zeros((1,3))
            for col in range(1,4):
                if self.R[row,col] == 1:
                    r_F[0,col-1]=reacc_F[pos]
                    r_N[0,col-1]=reacc_N[pos]
                    pos+=1
            temp_rf=np.insert(r_F,0,self.R[row,0])
            temp_rn=np.insert(r_N,0,self.R[row,0])
            R_F[row,:]=temp_rf
            R_N[row,:]=temp_rn

        self.Reac=np.zeros((self.R.shape[0],7))
        self.Reac[:,:4]=R_F
        self.Reac[:,4:]=R_N[:,1:4]

        #Creamos vector S
        self.S=np.zeros((self.E.shape[0],2))
        S_fuerza=np.zeros((self.E.shape[0],1))
        S_temp=np.zeros((self.E.shape[0],1))
        S_F=np.zeros((2,2))
        S_N=np.zeros((2,2))
        for elem in range(self.E.shape[0]):
            A=elementos_Dict[elem]['A']
            E=elementos_Dict[elem]['E']
            L=elementos_Dict[elem]['L']
            cof_dila=elementos_Dict[elem]['cof_dila']
            Transf=elementos_Dict[elem]['T']
            nlocal=elementos_Dict[elem]['nlocal']
            Ext=elementos_Dict[elem]['Ext']

            v_F=np.dot(Transf,np.dot(Ext,vector_U_F))
            v_N=np.dot(Transf,np.dot(Ext,vector_U_N))


            S_F=(A*E/L)*np.dot(np.array([[1,-1],[-1,1]]),v_F)
            S_N=(A*E/L)*np.dot(np.array([[1, -1],[-1, 1]]),v_N) + nlocal

            S_fuerza[elem,0]=(A*E/L)*np.dot(np.array([[1,-1]]),v_F)
            S_temp[elem,0]=(A*E/L)*np.dot(np.array([[1,-1]]),v_N)-A*E*self.dT[elem]*cof_dila

            self.S[elem,0]=S_fuerza[elem,0]
            self.S[elem,1]=S_temp[elem,0]

        return self.S,self.Reac,self.U

    def graficar(self,opcion='estructura',dim=2,escala=10.0):
        #obtencion de coordenadas de nodos
        if dim==2:
            if opcion=='estructura':
                nodos_c=[]
                codes = [Path.MOVETO]
                for elem in range(self.E.shape[0]):
                    ni=self.E[elem,0].astype(int)-1
                    nj=self.E[elem,1].astype(int)-1

                    nodos_c.append((self.C[ni,0],self.C[ni,1]))
                    nodos_c.append((self.C[nj,0],self.C[nj,1]))
                    if elem!= 0:
                        codes.append(Path.LINETO)
                        codes.append(Path.MOVETO)
                codes.append(Path.LINETO)
                path = Path(nodos_c, codes)
                fig = plt.figure()
                ax = fig.add_subplot(111)
                patch = patches.PathPatch(path,fill=False, lw=2, joinstyle='round')
                ax.add_patch(patch)
                ax.set_xlim(-10,np.max(self.C[:,0])+10)
                ax.set_ylim(-5,np.max(self.C[:,1])+10)
                plt.show()
            elif opcion=='deformada':

                nodos_c=[]
                nodos_def=[]
                codes=[]
                codes_def = [Path.MOVETO]
                codes = [Path.MOVETO]
                for elem in range(self.E.shape[0]):
                    ni=self.E[elem,0].astype(int)-1
                    nj=self.E[elem,1].astype(int)-1
                    cix=self.C[ni,0]+(self.U[ni,0]+self.U[ni,3])*float(escala)
                    ciy=self.C[ni,1]+(self.U[ni,1]+self.U[ni,4])*float(escala)
                    cjx=self.C[nj,0]+(self.U[nj,0]+self.U[nj,3])*float(escala)
                    cjy=self.C[nj,1]+(self.U[nj,1]+self.U[nj,4])*float(escala)
                    nodos_c.append((self.C[ni,0],self.C[ni,1]))
                    nodos_c.append((self.C[nj,0],self.C[nj,1]))
                    nodos_def.append((cix,ciy))
                    nodos_def.append((cjx,cjy))
                    if elem!= 0:
                        codes_def.append(Path.LINETO)
                        codes_def.append(Path.MOVETO)
                        codes.append(Path.LINETO)
                        codes.append(Path.MOVETO)
                codes_def.append(Path.LINETO)
                codes.append(Path.LINETO)
                path = Path(nodos_c, codes)
                path_2 = Path(nodos_def, codes_def)
                fig = plt.figure()
                ax = fig.add_subplot(111)
                patch_1 = patches.PathPatch(path,fill=False, lw=2)
                patch_2 = patches.PathPatch(path_2,fill=False, lw=1, color='r',linestyle='--' )
                ax.add_patch(patch_1)
                ax.add_patch(patch_2)
                ax.set_xlim(-10,np.max(self.C[:,0])+10)
                ax.set_ylim(-5,np.max(self.C[:,1])+10)
                plt.show()