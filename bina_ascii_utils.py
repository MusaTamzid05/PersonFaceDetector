import binascii

def four_char_to_int(string):
    return int(binascii.hexlify(string.encode()) , 16)

def int_to_four_chars(integer):
    return binascii.unhexlify(format(integer , "x")).decode()
