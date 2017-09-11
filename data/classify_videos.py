import os

cwd = os.getcwd()
path = os.path.join(cwd, 'trainset')

fid = open(os.path.join(path, 'train.txt'))
f_bdg = open(os.path.join(path, 'bridge.txt'), 'w+')
f_cty_etr = open(os.path.join(path, 'city_entry.txt'), 'w+')
f_cty_ext = open(os.path.join(path, 'city_exit.txt'), 'w+')
f_rd_bmp = open(os.path.join(path, 'road_bump.txt'), 'w+')
f_scrn_wprs = open(os.path.join(path, 'screen_wipers.txt'), 'w+')
f_zbr = open(os.path.join(path, 'zebra.txt'), 'w+')

for line in fid:
    filename = line.split()[0]
    solution = line.split()[1]
    bridge = int(solution[0])
    city_entry = int(solution[1])
    city_exit = int(solution[2])
    road_bump = int(solution[3])
    screen_wipers = int(solution[4])
    zebra = int(solution[5])
    print filename
    if bridge:
        f_bdg.write(filename + ' ')
    if city_entry:
        f_cty_etr.write(filename + ' ')
    if city_exit:
        f_cty_ext.write(filename + ' ')
    if road_bump:
        f_rd_bmp.write(filename + ' ')
    if screen_wipers:
        f_scrn_wprs.write(filename + ' ')
    if zebra:
        f_zbr.write(filename + ' ')
