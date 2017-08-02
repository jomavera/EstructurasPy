# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 17:44:35 2017

@author: Jose Manuel
"""
from Reticulado import *

ejemplo=Reticulado()

ejemplo.cargar_csv(directorio='datos/ejemplo_1')

S,R,U=ejemplo.rigidez()

ejemplo.graficar(opcion='deformada',escala=20)