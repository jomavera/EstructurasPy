# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 17:01:26 2017

@author: Jose Manuel
"""

from Reticulado import *

ejemplo=Reticulado()

ejemplo.cargar_csv(directorio='datos/ejemplo_2')

S,R,U=ejemplo.analizar()

ejemplo.graficar(opcion='deformada',escala=18,guardar='ejemplo2')