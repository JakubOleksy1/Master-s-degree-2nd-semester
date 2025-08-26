import csv
import math

#################################
#  name   | number of points    #
#  ------ | ------------------- #
#  tiny   |      10             #
#  small  |      30             #
#  medium |      100            #
#  large  |      1000           #
#################################
name = 'bigboi'

def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def transform_data(input_file, output_file):
    points = []

    with open(input_file, 'r') as infile:
        reader = csv.reader(infile)
        for row in reader:
            x, y = map(float, row)
            points.append((x, y))

    with open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        num_points = len(points)
        writer.writerow([num_points])
        writer.writerow(['path', 'point1x', 'point1y', 'point2x', 'point2y', 'distance'])

        for i in range(len(points)):
            x1, y1 = points[i]
            for j in range(i + 1, len(points)):
                x2, y2 = points[j]
                z = calculate_distance(x1, y1, x2, y2)
                writer.writerow([f'{i+1}-{j+1}', x1, y1, x2, y2, z])
                writer.writerow([f'{j+1}-{i+1}', x2, y2, x1, y1, z])

    return points

input_file = f'Supercomputer/{name}.csv'
output_file = f'Supercomputer/transformed_data_{name}.csv'

points = transform_data(input_file, output_file)

print(f"Data has been transformed and saved to {output_file}")