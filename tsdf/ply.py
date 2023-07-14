import numpy as np
import os


class Ply(object):
    """Class to represent a ply in memory, read plys, and write plys.
    """

    def __init__(self, ply_path=None, triangles=None, points=None, normals=None, colors=None):
        """Initialize the in memory ply representation.

        Args:
            ply_path (str, optional): Path to .ply file to read (note only
                supports text mode, not binary mode). Defaults to None.
            triangles (numpy.array [k, 3], optional): each row is a list of point indices used to
                render triangles. Defaults to None.
            points (numpy.array [n, 3], optional): each row represents a 3D point. Defaults to None.
            normals (numpy.array [n, 3], optional): each row represents the normal vector for the
                corresponding 3D point. Defaults to None.
            colors (numpy.array [n, 3], optional): each row represents the color of the
                corresponding 3D point. Defaults to None.
        """
        super().__init__()
        # TODO: If ply path is None, load in triangles, point, normals, colors.
        #       else load ply from file. If ply_path is specified AND other inputs
        #       are specified as well, ignore other inputs.
        if ply_path is None:
            self.triangles = triangles
            self.points = points
            self.normals = normals
            self.colors = colors
        else:
            self.read(ply_path)

        # TODO: If normals are not None make sure that there are equal number of points and normals.

        if self.normals is not None:
            assert self.points.shape[0] == self.normals.shape[0], 'Number of points and normals must be equal.'

        # TODO: If colors are not None make sure that there are equal number of colors and normals.

        if self.colors is not None:
            assert self.points.shape[0] == self.colors.shape[0], 'Number of points and colors must be equal.'
        

    def write(self, ply_path):
        """Write mesh, point cloud, or oriented point cloud to ply file.

        Args:
            ply_path (str): Output ply path.
        """
        #Write header depending on existance of normals, colors, and triangles.
        with open(ply_path, 'w') as f:

          
            f.write("ply\nformat ascii 1.0\nelement vertex ")
            #element vertex 0 if array is empty or None
            if np.shape(self.points)[0] == 0 or self.points is None:
                f.write("0\n")
            else:
                f.write(str(np.shape(self.points)[0]) + "\n")
                f.write("property float x\nproperty float y\nproperty float z\n")
                if self.normals is not None:
                    f.write("property float nx\nproperty float ny\nproperty float nz\n")
                if self.colors is not None:
                    f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            
            if self.triangles is not None:
                f.write("element face " + str(np.shape(self.triangles)[0]) + "\nproperty list uchar int vertex_indices\n")
            
            f.write("end_header\n")

            if self.points is not None:
                for i in range(np.shape(self.points)[0]):
                    f.write(str(self.points[i][0]) + " " + str(self.points[i][1]) + " " + str(self.points[i][2]) + " ")
                    if self.normals is not None:
                        f.write(str(self.normals[i][0]) + " " + str(self.normals[i][1]) + " " + str(self.normals[i][2]) + " ")
                    if self.colors is not None:
                        f.write(str(self.colors[i][0]) + " " + str(self.colors[i][1]) + " " + str(self.colors[i][2]) + " ")
                    f.write("\n")

            if self.triangles is not None:
                for i in range(np.shape(self.triangles)[0]):
                    f.write("3 " + str(self.triangles[i][0]) + " " + str(self.triangles[i][1]) + " " + str(self.triangles[i][2]) + "\n")

    def read(self, ply_path):
        """Read a ply into memory.

        Args:
            ply_path (str): ply to read in.
        """
        #Read in ply.
        assert(os.path.exists(ply_path)), "File does not exist."

        with open(ply_path, 'r') as f:
            assert "ply" in f.readline(), "File is not a ply."

            line = f.readline()

            #number of vertices
            while "element vertex" not in line.rstrip('\n'):
                line = f.readline().rstrip('\n')
            num_points = int(line.split(" ")[2])

            self.points, self.normals, self.colors = np.zeros((num_points, 3), float), np.zeros((num_points, 3), float), np.zeros((num_points, 3), int)

            #check if x, y, z properties are in the header, before "end_header"
            while "x" not in line.rstrip('\n') and "y" not in line.rstrip('\n') and "z" not in line.rstrip('\n'):
                line = f.readline().rstrip('\n')
                if "end_header" in line:
                    Exception("x, y, z properties not found in header.")
            
            #check if nx, ny, nz properties are in the header, before "red", "green", "blue", or "end_header"
            while "nx" not in line.rstrip('\n') and "ny" not in line.rstrip('\n') and "nz" not in line.rstrip('\n'):
                if "red" in line or "green" in line or "blue" in line or "end_header" in line:
                    self.normals = None
                    break
                line = f.readline().rstrip('\n')
            while "red" not in line.rstrip('\n') and "green" not in line.rstrip('\n') and "blue" not in line.rstrip('\n'):
                if "end_header" in line:
                    self.colors = None
                    break
                line = f.readline().rstrip('\n')
            
            #number of faces
            num_triangles = 0
            while line.rstrip('\n') != "end_header":
                line = f.readline().rstrip('\n')
                if "element face" in line:
                    num_triangles = int(line.split(" ")[2])
            

            #lines after header
            lines = f.readlines()

            for i in range(num_points):
                line = lines[i].split(" ")
                self.points[i] = [float(line[0]), float(line[1]), float(line[2])]
                #if normals are not None, add them to the array
                if self.normals is not None:
                    self.normals[i] = [float(line[3]), float(line[4]), float(line[5])]

                if self.colors is not None and self.normals is None:
                    self.colors[i] = [int(line[3]), int(line[4]), int(line[5])]
   
                elif self.colors is not None and self.normals is not None:
                    self.colors[i] = [int(line[6]), int(line[7]), int(line[8])]
            
            if num_triangles > 0:
                self.triangles = np.zeros((num_triangles, 3), int)
                for i in range(num_triangles):
                    line = lines[num_points + i].split(" ")
                    self.triangles[i] = [int(line[1]), int(line[2]), int(line[3])]
            else:
                self.triangles = None

        if self.normals is not None:
            assert self.points.shape[0] == self.normals.shape[0], 'Number of points and normals must be equal.'
        if self.colors is not None:
            assert self.points.shape[0] == self.colors.shape[0], 'Number of points and colors must be equal.'

            
