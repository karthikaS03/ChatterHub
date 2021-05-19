import pyshark
import json
from datetime import datetime
import sys
from pprint import pprint
from elasticsearch import Elasticsearch
import datetime as dt

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
				  "type" :{ "type": "keyword" },
				  "description" :{ "type": "keyword" },
				  "frame": {
				      "properties": {
					
					"time":{"type":"date", "format": "yyyy-MM-dd HH:mm:ss.SSSSSS||yyyy-MM-dd||epoch_millis"}
					
				      }
				    }
			  
			}
        	}
    	}
}


es = Elasticsearch(
    ['localhost'],
    port=9200

)
cap = []

with open('../json/logger.json') as f:
    cap = json.load(f)

packets=[]

for c in cap:
	packet={}
	packet["type"]="hub_log"
	packet["frame"]={}
	print(c['Date/Time'])
	packet["frame"]["time"]= (datetime.strptime(c['Date/Time'], '%m/%d/%Y %H:%M:%S.%f')+ dt.timedelta(hours=4) ).strftime('%Y-%m-%d %H:%M:%S.%f')
	packet["description"] = c['Device']+' '+c['Event Name']+' '+c['Description']
	packet["description"] =packet["description"].encode("utf-8")
	packets.append(packet)

es.indices.delete('packets_bulb_onoff_10sgap_hub')

es.indices.create(index = 'packets_bulb_onoff_10sgap_hub', body = request_body)
for ind,packet in enumerate(packets):
	print(ind)
	print (packet)
 	res = es.index(index='packets_bulb_onoff_10sgap_hub', doc_type='pcap_file', body=packet)
	print(res['result'])
	


