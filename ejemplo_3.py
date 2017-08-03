# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 20:35:08 2017

@author: Jose Manuel Vera Aray
"""

from Reticulado import *

ejemplo=Reticulado(dim=3)

ejemplo.cargar_csv(directorio='datos/ejemplo_3')

S,R,U=ejemplo.analizar()

ejemplo.graficar(opcion='deformada',escala=50,dim=3,guardar='ejemplo3')