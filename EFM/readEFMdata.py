
import struct

def read_binary_file(file_path):
    try:
        with open(file_path, 'rb') as file:
            binary_content = file.read()
            return binary_content
    except IOError as e:
        print(f"An error occurred while reading the file: {e}")
        return None

def interpret_binary_data(binary_data):
    # Assuming the binary file contains a sequence of 4-byte integers
    num_integers = len(binary_data) // 4
    integers = struct.unpack(f'{num_integers}i', binary_data)
    return integers

def main():
    file_path = '/data2/white/DATA/PROJ_LEE/IOP_2/EFMDATA/EFM_20221119_0252_IOP2_WestSmithville_GRAUPEL'
    binary_data = read_binary_file(file_path)
    
    if binary_data:
        interpreted_data = interpret_binary_data(binary_data)
        print("Interpreted data (as integers):")
        print(interpreted_data)
    else:
        print("No data read from the file.")

if __name__ == "__main__":
    main()

