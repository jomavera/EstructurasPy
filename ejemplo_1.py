# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 17:44:35 2017

@author: Jose Manuel
"""
from Reticulado import *

ejemplo=Reticulado()

ejemplo.cargar_csv(directorio='datos/ejemplo_1')

S,R,U=ejemplo.analizar()

ejemplo.graficar(opcion='deformada',escala=3,guardar='ejemplo1')