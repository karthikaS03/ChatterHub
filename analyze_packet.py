from sklearn.feature_extraction.text import CountVectorizer
import json
import pandas as pd
from datetime import datetime,timedelta
import datetime as dt_time
import calendar 
import pytz

HUB_IP = '10.42.0.55'
start_frame_time =0
end_frame_time =0
filtered_packets=[]
features = []

valid_event_values = set(['on','off','active','inactive','open','closed','pushed','held','locked','unlocked','online','offline','hubDisconnected'])

'''
Filters logs from server for the currently analyzed packets' timeframe
'''
def filter_hub_logs():
	global start_frame_time
	global end_frame_time	
	df = pd.read_csv('/home/sk-lab/Desktop/IoTproject/hub_logs/logger.csv')
	local = pytz.timezone('US/Eastern')
	start_time_local = local.localize(start_frame_time)
	print start_time_local
	hours_to_add=0
	if bool(start_time_local.dst()):
		hours_to_add = 4
	else:
		hours_to_add = 5
	df['timestamp'] = pd.to_datetime(df['Date/Time'],unit='ns')
	df['timestamp'] +=  pd.to_timedelta(hours_to_add, unit='h')

	start_frame_time = start_frame_time - timedelta(seconds=5)
	end_frame_time = end_frame_time + timedelta(seconds=60)
	

	df_filtered = df[(df.timestamp >= start_frame_time) & (df.timestamp <= end_frame_time)]
	#df_filtered.to_csv('/home/sk-lab/Desktop/IoTproject/hub_logs/filtered.csv', encoding='utf-8')
	#print zip(df_filtered['timestamp'], df_filtered['Event Name'])
	return  df_filtered

'''
Group packets based on their time difference
'''
def group_packets():
	global filtered_packets

	sequences=[]
	sequence=''
	times=[]

	l = len(filtered_packets)	
	time_delta =10

	for i in range(l-1):
		diff  = filtered_packets[i]['frame']['time']-filtered_packets[i-1]['frame']['time'] 		
		
		'''
		print filtered_packets[i-1]['frame']['time']
		print filtered_packets[i]['frame']['time']
		print filtered_packets[i+1]['frame']['time']		
		print diff.total_seconds()
		#print diff2.total_seconds()
		#print abs(diff.total_seconds() - diff2.total_seconds()) 
		print filtered_packets[i]['frame']['len']
		print '--------------------------------'
		'''
		
		#if the time difference is greater than delta, then it denotes start of a new command 
		if diff.total_seconds() <= time_delta:			
			sequence += filtered_packets[i]['frame']['len']+' '
			###times.append(calendar.timegm(filtered_packets[i]['frame']['time'].timetuple()))
			times.append(filtered_packets[i]['frame']['time'])
				
		else:			
			if len(times)==0:
				times.append(filtered_packets[i]['frame']['time'])
			###sequences.append(( sum(times)/len(times),sequence))
			sequences.append((times[0],sequence ))
			times = [filtered_packets[i]['frame']['time']]	
			sequence=filtered_packets[i]['frame']['len']+' '
	
	return sequences

'''
Group server logs based on their time difference
'''
def group_server_logs(df):
	rows= df.shape[0]
	sequence={}
	sequences=[]
	times =[] 
	time_delta = 10
	#print df['Event Name']
	for ind in range(rows):			
		diff  = (datetime.now()-df.iloc[ind]['timestamp']) if ind==rows-1 else (df.iloc[ind]['timestamp']-df.iloc[ind-1]['timestamp'])	
		dev_key = df.iloc[ind]['Device'].replace(' ','_')
		event_key = df.iloc[ind]['Event Name'].replace(' ','_')
		val = df.iloc[ind]['Event Value']
	
		if diff.total_seconds() <= time_delta :			
			if dev_key in sequence:				
				sequence[dev_key].append({'event':event_key ,'value':val})
			else:
				#sequence[dev_key] = {event_key : val}
				sequence[dev_key]=[{'event':event_key ,'value':val}]
			####times.append( calendar.timegm(df.iloc[ind]['timestamp'].timetuple()))
			times.append(df.iloc[ind]['timestamp'])
		else:			
			if len(times)==0:
				times.append(df.iloc[ind]['timestamp'])
			#sequences.append((sum(times)/len(times),sequence ))
			sequences.append((times[0],sequence ))
			
			times = [df.iloc[ind]['timestamp']]
			sequence = {dev_key:[{'event':event_key ,'value':val}]}
			
	
	return sequences
	
	
'''
Filters packets with data payload and discards acks
'''
def filter_packets(path, file_name):
	global start_frame_time
	global end_frame_time
	global filtered_packets
	global HUB_IP
	packets=[]
	doc = ''
	with open(path+file_name, 'r') as outfile:
     		packets = json.load(outfile)

	for packet in packets:
		if 'tcp' in packet:
			if packet['tcp']['flags']=='0x00000018' and packet.get('ssl')!=None and packet['ssl'].get('app_data')!=None:				
				doc = doc + str((int(packet['frame']['len']) / 100)*100) + ' '
				p = packet
				p['frame']['time'] = datetime.strptime(p['frame']['time'],'%Y-%m-%d %H:%M:%S.%f')
				p['frame']['mod_len']=str((int(packet['frame']['len']) / 100)*100) 
				p['frame']['len_direction'] = str(packet['frame']['len']) + '_' + ('hub' if packet['ip']['src'] == HUB_IP else 'server' )
				filtered_packets.append(p)
			

	start_frame_time =filtered_packets[0]['frame']['time']
	l = len(filtered_packets)
	end_frame_time = filtered_packets[l-1]['frame']['time']
	
	return doc


def get_packet_features():
	global filtered_packets
	global features
	time_delta = 20
	if filtered_packets:
		offset = filtered_packets[0]['frame']['time']
		for packet in  filtered_packets:
			diff = packet['frame']['time'] - offset
			if diff.total_seconds() >= time_delta:
				offset = packet['frame']['time']
				diff = 0
			feature = {					
						'packet_source'   : packet['frame']['len_direction'].split('_')[1],
						'frame_time'	  : packet['frame']['time'],
						'frame_length'    : packet['frame']['len'],
						'packet_length'   : packet['ip']['len'],
						'offset_time'	  : diff.total_seconds() if diff!=0 else 0.0
			}
			features.append(feature)
	

def process_mappings(command,seq):
	commands = []
	
	for device, events in command.items():
		for event in events:
			val = 'XXX'
			command= ''
			if event['value'] in valid_event_values:
				val = event['value']
			command = device#+'_'+event['event']+'_'+val
			all_events.add(command)
			commands.append(command)
	
	return (' '.join(commands),seq)

'''
Maps groups of packets to the groups of cluster logs
'''
def generate_mappings(path,file_name):

	global filtered_packets
	filtered_packets=[]
	
	filter_packets(path,file_name)
	get_packet_features()

	seq_1 = group_packets()
	seq_2= group_server_logs(filter_hub_logs())
	command_seq=[]
	
	'''
	for (time,seq) in seq_2:
		for i in range(len(seq_1)):
			t2,_= seq_1[i]
			#loop until two groups with minimum time difference between them is found and stop searching for more groups
			if abs(t2-time)<=20:	
				
				if command_seq:
					prev_cmd,prev_seq = command_seq[-1]
					prev_seq = prev_seq + ' '.join([seq_1[x][1].rstrip() for x in range(0,i)])
					command_seq[-1] = (prev_cmd,prev_seq) 		
					
				command_seq.append(process_mappings(seq,seq_1[i][1]))				
				seq_1 = seq_1[i+1:]
				#seq_1.pop(i)
				break
	'''
	for (time,seq) in seq_2:
		mapped=[]
		per_seq = ''
		for i in range(len(seq_1)):
			t2,_= seq_1[i]
			#loop until two groups with minimum time difference between them is found and stop searching for more groups
			if abs((t2-time).total_seconds())<=20:	
				'''
				if command_seq:
					prev_cmd,prev_seq = command_seq[-1]
					prev_seq = prev_seq + ' '.join([seq_1[x][1].rstrip() for x in range(0,i)])
					command_seq[-1] = (prev_cmd,prev_seq) 		
				'''	
				per_seq +=seq_1[i][1]
				mapped.append(i)
			elif len(per_seq)>0:
				break
		if per_seq=='' and len(command_seq)>0:
			cmd,_ = process_mappings(seq,per_seq)
			command_seq[-1] = (command_seq[-1][0]+' '+cmd,command_seq[-1][1])	
		else:
			if "115 121 " in per_seq:
				per_seq = per_seq.replace("115 121 ","")
			command_seq.append(process_mappings(seq,per_seq))
		if mapped:
			seq_1=seq_1[0:mapped[0]]+seq_1[mapped[-1]+1:]
		
	'''
	print '************unmapped packets*******************'
	print seq_1
	'''
	return command_seq

import os 
commands = []
sequences = []
command_seq = []
command_seq2={}
all_events = set()
path = '/home/sk-lab/Desktop/IoTproject/json/HMM/'

for file in  os.listdir(path):
	if file!='logger.json':
		for com_seq in generate_mappings(path,file):
			commands.append(com_seq[0])#+'\n'+com_seq[1]+'\n------------------------------------------------------------------------------------')
			sequences.append(com_seq[1])
			com = list(set(com_seq[0].split(' ')))[0]
			if command_seq2.get(com)!=None:
				command_seq2[com] = command_seq2[com] + com_seq[1]
			else:
				command_seq2[com]=com_seq[1]
			command_seq.append({'device':com_seq[0],'sequence':com_seq[1].rstrip().split(' ')})
			

'''
with open('../results/command_mapping_train.txt','w') as out:
	out.write('\n'.join(commands))

with open('../results/seq_mapping_train.txt','w') as out:
	out.write('\n'.join(sequences))

with open('../results/events_train.txt','w') as out:
	out.write('\n'.join(all_events))


df_features = pd.DataFrame(features)
df_features.to_csv('../results/features.csv')
'''

with open('command_seq_mapping_HMM_input_vect_test.txt','w') as out:
	 out.write(json.dumps(command_seq2, sort_keys=False,indent=4,encoding="utf-8",ensure_ascii=False))



with open('command_seq_mapping_HMM_input_test.txt','w') as out:
	 out.write(json.dumps(command_seq, sort_keys=False,indent=4,encoding="utf-8",ensure_ascii=False))





'''
def map_commands_to_packets(df):
	global filtered_packets
	l = len(filtered_packets)
	count=0
	#print df
	sequences=[]
	rows= df.shape[0]
	for ind in range(rows):
		sequence=''
		print '************************************************************'
		print df.iloc[ind]['Event Name']
		print df.iloc[ind]['Event Value']
		for i in range(count,l):
			diff  = df.iloc[ind]['timestamp']-filtered_packets[i]['frame']['time'] 			
			diff2 =  (datetime.now()-filtered_packets[i]['frame']['time']) if ind==rows-1 else (df.iloc[ind+1]['timestamp']-filtered_packets[i]['frame']['time'])
			
			print df.iloc[ind]['timestamp']
			print filtered_packets[i]['frame']['time']
			print df.iloc[ind+1]['timestamp']
			print filtered_packets[i]['frame']['time']
			print diff.total_seconds()
			print diff2.total_seconds()
			print abs(diff.total_seconds()) < abs(diff2.total_seconds()) 
			print filtered_packets[i]['frame']['len']
			print '--------------------------------'
			
			if  abs(diff.total_seconds()) <= abs(diff2.total_seconds())  :
				#print filtered_packets[i]['frame']['time']
				sequence += filtered_packets[i]['frame']['mod_len']+' '
			else:	
				count=i			
				break
		print sequence
		print df.iloc[ind]['timestamp']
		print df.iloc[ind]['Event Name']
		print df.iloc[ind]['Event Value']
		sequences.append((df.iloc[ind]['Event Name'],sequence))
	print sequences

Ignore this method!!!

def get_frequency_count():
	corpus = []
	corpus.append(filter_packet_lengths())
	vectorizer = CountVectorizer(ngram_range=(2,5))
	X = vectorizer.fit_transform(corpus)
	vocab = vectorizer.vocabulary_
	count_values = X.toarray().sum(axis=0)
	counts = sorted([(count_values[i],k) for k,i in vocab.items()], reverse=True)
	#print(counts[:30])
'''
