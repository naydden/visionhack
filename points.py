import os

# NOM DEL FITXER RESULTANT
output_filename = 'output.txt'

video_folder = '../validationset/'


################     --   INSTRUCTIONS   --      ################
### Write the path to the file relative tot the current directory
### Write the column that needs to be extracted from the file

input_files = [
  ('maxmin_dif.txt', 1), # FITXER BRIDGE
  (None, 2), # FITXER CITY ENTER
  (None, 3), # FITXER CITY EXIT
  ('test.txt', 1), # FITXER ROAD BUMP
  ('maxmin_dif.txt', 5), # FITXER SCREEN WIPERS
  ('test.txt', 2) # FITXER ZEBRA
]


# Actual working code
cwd = os.getcwd()

data = []

ind = 0
for f in os.listdir(os.path.join(cwd, video_folder)):
    if f[-4:] == '.avi':
        data.append(f + ' ')
        ind = ind + 1

for fil in input_files:
    if fil[0] == None:
        ind = 0
        for line in data:
            data[ind] = data[ind] + '0'
            ind = ind + 1
        
    else:
        path = os.path.join(cwd, fil[0])
        position = fil[1] - 1
        
        fid = open(path, 'r')
        
        lines = fid.read().split('\n')
        
        ind = 0
        for line in lines:
            print 'Line' + str(ind) + ' ' + line
            data[ind] = (data[ind] or '') + str(line[21:][position])
            ind = ind + 1

fid = open(os.path.join(cwd, output_filename), 'w+')

for line in data:
    fid.write(line + '\n')
