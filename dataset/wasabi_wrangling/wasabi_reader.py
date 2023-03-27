import csv

rows_to_read = 2
filename = 'wasabi_songs.csv'

with open(filename) as f:
    csv_reader = csv.reader(f, delimiter=',')

    l = 0

    for row in csv_reader:
        if l < rows_to_read:
            print(len(row))
            for item in row:
                print(item)
            l += 1
        else:
            # done
            break
