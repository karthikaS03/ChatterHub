packet_dict = {
'103':'p1',
'111':'p2',
'1185':'p3',
'120':'p4',
'121':'p5',
'122':'p6',
'129':'p7',
'132':'p8',
'138':'p9',
'141':'p10',
'145':'p11',
'146':'p12',
'223':'p13',
'284':'p14',
'285':'p15',
'304':'p16',
'305':'p17',
'306':'p18',
'342':'p19',
'414':'p20',
'415':'p21',
'416':'p22',
'417':'p23',
'418':'p24',
'419':'p25',
'420':'p26',
'421':'p27',
'425':'p28',
'426':'p29',
'427':'p30',
'434':'p31',
'435':'p32',
'436':'p33',
'438':'p34',
'439':'p35',
'535':'p36',
'559':'p37',
'744':'p38'
}


# convert a space-separated list of packet sizes to a list of string packet size
def space_str_to_list(str_space):
    ret = [str(i) for i in str_space.split()]
    print("string in list form: {}".format(ret))
    return ret


# convert a comma-separated list of packet sizes to a list of string packet size
def comma_str_to_list(str_space):
    ret = [str(i) for i in str_space.split(',')]
    print("string in list form: {}".format(ret))
    return ret


# replace string packet sizes with packet object (enclosed by quotes)
def packet_to_obj(packet_list):
    for n, i in enumerate(packet_list):
        # print('{}, {}'.format(n, i))
        if i in packet_dict:
            packet_list[n] = packet_dict[i]
        else:
            print('{} to be removed'.format(i))
            del packet_list[n]
    print("final object list: {}".format(packet_list))
    return packet_list


# sequences = [
#     {'device_name': 'Kwikset_10-Button_Deadbolt', 'packets':
#     [['145', '129', '138', '121', '146', '122'],
#      ['342', '1185', '138', '121'],
#      ['138', '121'],
#      ['138', '121'],
#      ['132', '111', '146', '120', '223', '103', '535', '141', '138', '121'],
#      ['145', '129', '138', '121', '146', '122'],
#      ['145', '129', '138', '121', '146', '122'],
#      ['138', '121'],
#      ['145', '129', '138', '121', '146', '122']]},
#     {'device_name': 'SYLVANIA_Smart_10Y_A19_TW', 'packets':
#         [['304', '425'],
#          ['304', '426'],
#          ['304', '427'],
#          ['304', '425'],
#          ['304', '426', '132', '111', '146', '120', '223', '103', '535', '141'],
#          ['285', '419', '421', '416', '418'],
#          ['284', '419', '420', '417', '416'],
#          ['285', '417', '419', '416', '416'],
#          ['285', '744', '417', '417'],
#          ['285', '418', '419', '416', '132', '111', '146', '120', '223', '103', '535', '141', '416'],
#          ['284', '420', '421', '417', '417'],
#          ['285', '419', '420', '415', '417'],
#          ['306', '418', '417'],
#          ['305', '416', '416'],
#          ['305', '416', '414'],
#          ['306', '417', '417'],
#          ['306', '417', '416']]},
#     {'device_name': 'OSRAM_LIGHTIFY_Dimming_Switch', 'packets':
#         [['438'],
#          ['438'],
#          ['439'],
#          ['439'],
#          ['438'],
#          ['342', '1185', '435'],
#          ['132', '111', '146', '120', '223', '103', '535', '141', '434'],
#          ['435'],
#          ['436'],
#          ['435'],
#          ['435'],
#          ['435'],
#          ['436'],
#          ['438']]},
#     {'device_name': 'SYLVANIA_SMART+_Smart_Plug', 'packets':
#         [['417'],
#          ['417'],
#          ['559', '417'],
#          ['416'],
#          ['417', '419']]}
# ]



# deadbolt_light = {'name':'deadbolt_light', 'list': '438,145,129,138,121,146,145,122,129,138,121,146,122,439,436,145,129,138,121,146,122'}
# bulb_light = {'name':'bulb_light', 'list': '285,419,420,417,417,438,285,419,772,417,436,438'}
# deadbolt_bulb = {'name':'deadbolt_bulb', 'list': '285,772,416,416,145,129,138,121,146,145,122,129,138,121,146,122,285,772,416,416'}
# deadbolt_light_test = {'name': 'deadbolt_light', 'list':'285,417,419,415,415,145,129,138,121,146,145,122,129,138,121,146,122,306,416,416,284,771,417,415,145,129,138,121,146,122,304,424,285,741,415,416,342,1185,305,425'}

# TO-DO: read sequence list from json object
# first remove commas or spaces, and switching packet sizes to packet objects. DONE
# TO-DO: some packet sizes don't switch to objects, must handle this issue by:
#   creating packet objects that handle all cases (training data must encapsulate most, if not all, of the packet sizes)
#   create dummy packet sizes, and handle accordingly in the hmm
#   remove from sequence altogether (current)
# TO-DO: remove packet objects enclosed with single quotes automatically
# TO-DO: When piped to pomegranate, read the list key in the object, since the object has both name and list



#  Steps:
#  obtain devices from the Wireshark readings
#  assign them to variables, and store the name of the device as well
#  convert the list of packet sizes to p objects (p objects in a dictionaryy above for conversion)
#  use the results in the HMM code, found in the jupyyter notebook

# smart plug
smart_plug1 = {'name': 'smart_plug1', 'list': '417 113'}
smart_plug2 = {'name': 'smart_plug2', 'list': '417 113'}
smart_plug3 = {'name': 'smart_plug3', 'list': '559 417 113'}
smart_plug4 = {'name': 'smart_plug4', 'list': '416 113'}
smart_plug5 = {'name': 'smart_plug5', 'list': '115 121 417 113 419 113'}

# bulb
bulb1 = {'name': 'bulb1', 'list': '306 113 417 113 416 113'}
bulb2 = {'name': 'bulb2', 'list': '307 113 417 113 416 113'}
bulb3 = {'name': 'bulb3', 'list': '306 113 417 113 416 113'}
bulb4 = {'name': 'bulb4', 'list': '306 113 417 113 416 113'}
bulb5 = {'name': 'bulb5', 'list': '306 113 417 113 417 113 419 113'}

# light
light1 = {'name': 'light1', 'list': '438 113 138 113 121 113'}
light2 = {'name': 'light2', 'list': '115 121 425 113 437 113'}
light3 = {'name': 'light3', 'list': '435 113'}
light4 = {'name': 'light4', 'list': '115 121 438 113'}
light5 = {'name': 'light5', 'list': '115 121 437 113 425 113'}

# deadbolt
deadbolt1 = {'name': 'deadbolt1', 'list': '132 111 146 120 223 103 535 141 115 121 138 113 121 113 421 113'}
deadbolt2 = {'name': 'deadbolt2', 'list': '132 111 146 120 223 103 535 141 138 113 121 113'}
deadbolt3 = {'name': 'deadbolt3', 'list': '115 121 138 113 121 113'}
deadbolt4 = {'name': 'deadbolt4', 'list': '121 113 138 113 121 113'}
deadbolt5 = {'name': 'deadbolt5', 'list': '115 121 138 113 121 113'}

device_str_list = [
    smart_plug1, smart_plug2, smart_plug3, smart_plug4, smart_plug5,
    bulb1, bulb2, bulb3, bulb4, bulb5,
    light1, light2, light3, light4, light5,
    deadbolt1, deadbolt2, deadbolt3, deadbolt4, deadbolt5
]

final_list = []

# TO-DO: i'm just taking the output from terminal, add the device names and return a list of objects, device name + list
for device in device_str_list:
    # str_list = comma_str_to_list(device['list'])
    str_list = space_str_to_list(device['list'])

    device['list'] = packet_to_obj(str_list)
    final_list.append(device)

print("final list:")
for device in final_list:
    print(device)
    print("")