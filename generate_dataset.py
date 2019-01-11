import json
import os

hub_segments_path = '../results/test/hub_segments/'
pcap_segments_path = '../results/test/pcap_segments/'

def generate_dataset(file_name, pcap_parameter, hub_parameters):
   pcap_segments = {}
   hub_segments = {}

   with open(pcap_segments_path+file_name, 'r') as out:
      pcap_segments = json.load(out)
   
   with open(hub_segments_path+file_name, 'r') as out:
      hub_segments = json.load(out)

   dataset = []

   for seq_no,seq in pcap_segments.items():
      segments = hub_segments[seq_no]
      seq_events = []
      
      for seg in segments:
         seq_events.append('_'.join([seg[param] for param in hub_parameters ]))
      
      seq_frame_details = ' '.join([frame[pcap_parameter] for frame in seq])
      seq_event_label = ' '.join(set(seq_events))
      dataset.append({'packet_sequence':seq_frame_details,'sequence_label': seq_event_label})
   
   return dataset
   



full_dataset =[]

for file in  os.listdir(hub_segments_path):
   
      #list of parameters can be found in files in pcap segments folder
      pcap_parameter = 'frame_length'
      #modify based on the parameters you would like for class labels //list found in hub segments
      hub_parameters = ['device','event','val']

      #dataset for each specific devices
      dataset = generate_dataset(file, pcap_parameter, hub_parameters)
      #dataset for all devices
      full_dataset = full_dataset + dataset

      with open('../results/dataset/'+file,'w') as out:
         out.write(json.dumps(dataset, sort_keys=False,indent=4,ensure_ascii=False, default=str))

with open('../results/dataset/all_devices_test.json','w') as out:
         out.write(json.dumps(full_dataset, sort_keys=False,indent=4,ensure_ascii=False, default=str))