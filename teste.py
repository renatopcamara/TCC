# coding: utf-8

import time
import datetime

def ataque():
	antes =0
	agora = datetime.datetime.now().strftime("%S")
	while True:
		if antes == 0:
			antes = datetime.datetime.now().strftime("%S")
			#print("Antes:", antes)
		if agora == antes: # nesse caso deve fazer append na lista
			agora = datetime.datetime.now().strftime("%S")
			#print("Agora = Antes == Agora:",agora ,"antes",antes)
		else: # nesse caso deve sair com o MAX da lista e zerar a lista
			antes = datetime.datetime.now().strftime("%S")
			#print("Agora <> Anter == Agora:",agora ,"antes",antes)
			#print( str(datetime.datetime.now().strftime("%H:%M:%S")))
			print(int(time.time()))
		time.sleep(0.5)
	

antes =0
agora = int(time.time())
intervalo = 4 # intervalo em segundos + 1
while True:
	if antes == 0:
		antes = int(time.time())
	if antes >= agora - intervalo: # nesse caso deve fazer append na lista
		agora = int(time.time())
	else: # nesse caso deve sair com o MAX da lista e zerar a lista
		antes = int(time.time())
		print(int(time.time()))
	#time.sleep(0.5)