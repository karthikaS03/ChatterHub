import pyshark
import json
from datetime import datetime
import sys
from pprint import pprint
from elasticsearch import Elasticsearch

request_body = {
	    "settings" : {
	        "number_of_shards": 5,
	        "number_of_replicas": 1
	    },
	"mappings":
    	{
		"pcap_file":
		{
		
			  "properties": 
			{
				  "frame": {
				      "properties": {
					"len": { "type": "long" },
					"number": { "type": "long" },
					"protocols": { "type": "keyword" },
					"time":{"type":"date", "format": "yyyy-MM-dd HH:mm:ss.SSSSSS||yyyy-MM-dd||epoch_millis"},
					"date":{"type":"date", "format": "yyyy-MM-dd"},
					"offset_shift":{"type":"long"},
					"time_epoch":{"type":"date", "format": "yyyy-MM-dd HH:mm:ss.SSSSSS||yyyy-MM-dd||epoch_millis"},
					"time_delta":{"type":"long"},
					"time_delta_displayed":{"type":"long"},
					"time_relative":{"type":"long"}
				      }
				    },
				    "ip": {
				      "properties": {
					"src": {  "type": "ip"},
					"dst": {  "type": "ip"}
				      }
				    },
				    "udp": {
				      "properties": {
					"srcport": { "type": "integer"},
					"dstport": { "type": "integer"}
				      }
				    }
			  
			}
        	}
    	}
}

def generate_pcap_json(path, file_name):

	es = Elasticsearch(
		['localhost'],
		port=9200

	)
	cap = pyshark.FileCapture(path+file_name)
	packets=[]
	
	'''
	duration=[]
	prev_time = datetime.strptime( datetime.utcfromtimestamp(float(cap[0].frame_info.get('time_epoch'))).strftime('%Y-%m-%d %H:%M:%S.%f'),'%Y-%m-%d %H:%M:%S.%f')
	for c in cap:
		frame = c.frame_info
		for l in c.layers:
			l_name = l.layer_name			
			if l_name=='tcp':
				if l.get('flags') == '0x00000018':
					curr_time = datetime.strptime( datetime.utcfromtimestamp(float(frame.get('time_epoch'))).strftime('%Y-%m-%d %H:%M:%S.%f'),'%Y-%m-%d %H:%M:%S.%f')
					duration.append(str((curr_time - prev_time).total_seconds()))
					packets.append(str(frame.get('len')))
					#prev_time = curr_time
		

	with open('/home/sk-lab/Desktop/IoTproject/results/'+file_name.split('.')[0]+'.txt', 'w') as outfile:
		outfile.write('],['.join(duration))
		outfile.write('\n'+ ','.join(packets))
	'''
	for c in cap:
		packet={}
		l = c.frame_info		
		l_name = l.layer_name
		packet['type']='pcap_log'
		packet[l_name]={}

		for field in l.field_names:
			packet[l_name][field] = l.get(field)		

		packet[l_name]['time']=datetime.utcfromtimestamp(float(l.get('time_epoch'))).strftime('%Y-%m-%d %H:%M:%S.%f')
		packet[l_name]['date']=datetime.utcfromtimestamp(float(l.get('time_epoch'))).strftime('%Y-%m-%d')	
		packet['description']=''
		for l in c.layers:
			l_name = l.layer_name
			packet[l_name]={}
			if l_name=='ip':
				packet['description'] = packet['frame']['len'] +' '+ l.get('src')
				packet['mod_len'] = int(round(int(packet['frame']['len'])))
			
			for field in l.field_names:
				if field!="":
					packet[l_name][field] = l.get(field)
		
		packets.append(packet)
	

	#es.indices.delete('packets_bulb_onoff_10sgap')
	'''
	es.indices.create(index = 'packets_pcap_multipupose_open_close_only', body = request_body)
	for ind,packet in enumerate(packets):
		es.index(index='packets_pcap_multipupose_open_close_only', doc_type='pcap_file', body=packet)
	'''	
	
	with open('/home/sk-lab/Desktop/IoTproject/json/test/'+file_name.split('.')[0]+'.json', 'w') as outfile:
		json.dump(packets, outfile, sort_keys = True, indent = 4,
				ensure_ascii = False)

path = '/home/sk-lab/Desktop/IoTproject/pcaps/test/'
import os 
for file in  os.listdir(path):
	print file
	generate_pcap_json(path, file)