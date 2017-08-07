# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 22:28:26 2017

@author: Jose Manuel Vera Aray
"""

from Marco2D import *

ejemplo=Marco2D(corte=1)

ejemplo.cargar_csv(directorio='datos/ejemplo_4')

S,R,U=ejemplo.analizar()

