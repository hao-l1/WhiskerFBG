import trimesh
import coacd
import os 
import matplotlib.pyplot as plt
from tqdm import tqdm

def visualize_mesh(mesh):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(mesh.vertices[:,0], mesh.vertices[:,1], mesh.vertices[:,2], triangles=mesh.faces, linewidth=0.2, antialiased=True)
    plt.show()

def visualize_parts(parts):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for vs, fs in parts:
        ax.plot_trisurf(vs[:,0], vs[:,1], vs[:,2], triangles=fs, linewidth=0.2, antialiased=True)
    plt.show()

def main(start_from=0):
    paths = sorted(os.listdir("models"))[start_from:]

    for path in tqdm(paths):    
        mesh = trimesh.load(f"models/{path}/google_16k/nontextured.ply", force="mesh")
        mesh = coacd.Mesh(mesh.vertices, mesh.faces)
        parts = coacd.run_coacd(mesh) # a list of convex hulls.

        save_path = f"models/{path}/google_16k/parts"
        os.system(f"mkdir -p {save_path}")

        for i, (vs, fs) in enumerate(parts):
            mesh = trimesh.Trimesh(vs, fs)
            mesh.export(f"{save_path}/parts_"+str(i)+".stl")
        
        tqdm.write(f"Finished {path}!")

if __name__ == "__main__":
    
    main()