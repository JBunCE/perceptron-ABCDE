import random
import threading
import time
import numpy as np
import pandas as pd
import pandastable as pdt
import random
import customtkinter as ctk
import matplotlib.pyplot as plt

from tkinter import PhotoImage
from PIL import Image, ImageTk
from matplotlib.animation import FuncAnimation, PillowWriter

'''
0.00001
0.00001
1000
'''

class Perceptron(ctk.CTk):
    def __init__(self) -> None:
        # UI
        super().__init__()

        self.title("Perceptron")

        self.geometry("1366x768")

        self.options_frame = ctk.CTkFrame(self)
        self.options_frame.pack(side="left", ipadx=10, ipady=10, fill="y", expand=False, padx=10, pady=10)

        self.learing_rate_label = ctk.CTkLabel(self.options_frame, text="Learning rate")
        self.learing_rate_label.pack(padx=10, pady=10)

        self.learing_rate_input = ctk.CTkEntry(self.options_frame)
        self.learing_rate_input.pack(padx=10, pady=10)

        self.error_rate_label = ctk.CTkLabel(self.options_frame, text="Error rate")
        self.error_rate_label.pack(padx=10, pady=10)

        self.error_rate_input = ctk.CTkEntry(self.options_frame)
        self.error_rate_input.pack(padx=10, pady=10)

        self.epoch_input_label = ctk.CTkLabel(self.options_frame, text="Epoch input")
        self.epoch_input_label.pack(padx=10, pady=10)

        self.epoch_input = ctk.CTkEntry(self.options_frame)
        self.epoch_input.pack(padx=10, pady=10)

        self.tables_frame = ctk.CTkFrame(self)
        self.tables_frame.pack(side="right", fill="y", ipadx=10, ipady=10, padx=10, pady=10)

        self.table = pdt.Table(self.tables_frame, dataframe=pd.DataFrame({
            "w0": [0],
            "w1": [0],
            "w2": [0],
            "w3": [0],
            "w4": [0]
        }))
        self.table.show()

        self.chart_frame = ctk.CTkFrame(self)
        self.chart_frame.pack(side="right", ipadx=10, ipady=10, fill="both", expand=True, padx=10, pady=10)

        self.start_button = ctk.CTkButton(self.options_frame, text="Start", command=self.start)
        self.start_button.pack()

        self.canvas = ctk.CTkCanvas(self.chart_frame, bg="black")
        self.canvas.pack(expand=True, fill="both")

        self.y_calc_per_epoch_animation_button = ctk.CTkButton(self.options_frame, text="Y Calc per epoch", command=lambda: self.clear_and_play_gif(f"./runs/{len(self.error_per_run)}/y_calc"))
        self.y_calc_per_epoch_animation_button.pack(padx=10, pady=10)

        self.error_per_epoch_animation_button = ctk.CTkButton(self.options_frame, text="Error per epoch", command=lambda: self.clear_and_play_gif(f"./runs/{len(self.error_per_run)}/error"))
        self.error_per_epoch_animation_button.pack(padx=10, pady=10)

        self.weights_per_epoch_animation_button = ctk.CTkButton(self.options_frame, text="Weights per epoch", command=lambda: self.clear_and_play_gif(f"./runs/{len(self.error_per_run)}/weighs"))
        self.weights_per_epoch_animation_button.pack(padx=10, pady=10)

        self.error_per_run_animation_button = ctk.CTkButton(self.options_frame, text="Error per run", command=lambda: self.clear_and_play_gif(f"./runs/{len(self.error_per_run)}/error_per_run"))
        self.error_per_run_animation_button.pack(padx=10, pady=10)

        self.aanimation_button = ctk.CTkButton(self.options_frame, text="Make charts animation", command=self.make_charts_animation)
        self.aanimation_button.pack(padx=10, pady=10)

        # Initialization
        # self.problem_df = pd.read_excel('data.xlsx', skiprows=1)
        self.problem_df = pd.read_csv('213500.csv', sep=';')
        print(self.problem_df.columns)

        self.problem_df.columns = self.problem_df.columns.str.strip()
        self.problem_df = self.problem_df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        
        self.yd = np.array(self.problem_df['y'].values.tolist())
        self.yd_p = self.yd
        print(self.yd)
        
        # self.n = (self.yd - self.yd.min()) / (self.yd.max() - self.yd.min())
        # self.yd = self.n

        print(self.yd)
                
        self.x = np.array(self.problem_df[["x1", "x2", "x3", "x4"]].values.tolist())
        
        # Normalize input data X to the range [-1, 1]
        # x_min = np.min(self.x, axis=0)
        # x_max = np.max(self.x, axis=0)
        # self.x = -1 + 2 * (self.x - x_min) / (x_max - x_min)
        
        self.b = 1
        self.w = np.random.rand(1, 5)
        self.learning_rate = 0.1
        
        self.y_calc = None
        self.error = None
        
        self.error_per_epoch = []
        self.y_calc_per_epoch = []
        self.weighs_per_epoch = []
        self.error_per_run = []

        self.after_id = None
        self.gif = False
        self.counter = 0

        self.lrates = []
        self.erates = []

    def clear_and_play_gif(self, gif_filename):
        if self.gif:
            if self.after_id is not None:
                self.after_cancel(self.after_id)

            # Clear the canvas
            self.canvas.delete("all")

            # Play the GIF
            self.play_gif(gif_filename + ".gif", 0)
        else:
            self.put_img(gif_filename + ".png")
    
    def put_img(self, gif_filename):
        img = PhotoImage(file=gif_filename)
        self.canvas.create_image(0, 0, anchor="nw", image=img)
        self.canvas.image = img

    def play_gif(self, gif_filename, frame):
        img = Image.open(gif_filename)
        gif_frames = []

        try:
            while True:
                gif_frames.append(ImageTk.PhotoImage(img.copy().convert('RGBA')))
                img.seek(len(gif_frames))  # Seek to the next frame
        except EOFError:
            pass  # Reached the end of the gif

        def update_frame(frame):
            if frame < len(gif_frames):
                self.canvas.create_image(0, 0, anchor="nw", image=gif_frames[frame])
                self.after_id = self.after(100, update_frame, frame + 1)
            else:
                self.canvas.delete("all")  # Delete the image when the GIF ends
                
        update_frame(frame)
        
    def start(self):
        self.error_per_epoch = []
        self.y_calc_per_epoch = []
        self.weighs_per_epoch = []

        epoch = 0

        self.learning_rate = float(self.learing_rate_input.get())
        self.error_rate = float(self.error_rate_input.get())

        self.b = 1
        self.w = np.random.rand(1, 5)
        self.error = None
        self.y_calc = None

        x = self.x
        x = np.column_stack(([self.b for i in range(x.shape[0])], x))
        while int(self.epoch_input.get()) > epoch:          
            u = np.dot(self.w, x.T)
            self.y_calc = self._f(u)
            
            self.error = self.y_calc - self.yd 
            
            self.y_calc_per_epoch.append(self.y_calc)
            self.weighs_per_epoch.append(self.w)
            self.error_per_epoch.append(np.linalg.norm(self.error))
       
            d_w = self.learning_rate * np.dot(self.error, x)
            self.w = self.w - d_w

            epoch += 1
            print("epoch: ", epoch)
            
            if np.linalg.norm(self.error) <= self.error_rate:
                print(f"\n error {np.linalg.norm(self.error)}")
                print(f"\n Final weights: {self.w}")
                print(f"\n Final output: {self.y_calc * (self.yd_p.max() - self.yd_p.min()) + self.yd_p.min()}")
                break

        print(f"\n error {np.linalg.norm(self.error)}")
        print(f"\n Final weights: {self.w}")
        print(f"\n Final output: {self.y_calc * (self.yd_p.max() - self.yd_p.min()) + self.yd_p.min()}")
        weighs_df = pd.DataFrame({
            "w0": [self.weighs_per_epoch[-1][0][0]],
            "w1": [self.weighs_per_epoch[-1][0][1]],
            "w2": [self.weighs_per_epoch[-1][0][2]],
            "w3": [self.weighs_per_epoch[-1][0][3]],
            "w4": [self.weighs_per_epoch[-1][0][4]]
        })
        self.table = pdt.Table(self.tables_frame, dataframe=weighs_df)
        self.table.show()

        self.error_per_run.append(self.error_per_epoch)
        self.lrates.append(self.learning_rate)
        self.erates.append(self.error_rate)

        self.make_charts()

    def make_charts(self):
        y_calc_fig = plt.figure(figsize=(11, 10))
        ax_y_calc = y_calc_fig.add_subplot(111)
        ax_y_calc.set_facecolor("black")
        ax_y_calc.plot([i for i in range(len(self.yd_p))], self.yd_p, label="yd", color="green")
        ax_y_calc.plot([i for i in range(len(self.yd_p))], self.y_calc[0], '--', label="y_calc", color="red")
        ax_y_calc.legend()
        ax_y_calc.set_title("Yd vs Y_calc")
        ax_y_calc.set_xlabel("Samples")
        ax_y_calc.set_ylabel("Values")
        
        #save fig
        y_calc_fig.savefig(f"./runs/{len(self.error_per_run)}/y_calc.png")

        error_fig = plt.figure(figsize=(11, 10))
        ax_error = error_fig.add_subplot(111)
        ax_error.plot([i for i in range(len(self.error_per_epoch))], self.error_per_epoch)
        ax_error.set_title("Error per epoch")
        ax_error.set_xlabel("Epoch")
        ax_error.set_ylabel("Error")

        #save fig
        error_fig.savefig(f"./runs/{len(self.error_per_run)}/error.png")

        weighs_per_epoch_fig = plt.figure(figsize=(11, 10))
        ax_weighs_per_epoch = weighs_per_epoch_fig.add_subplot(111)
        ax_weighs_per_epoch.plot([i for i in range(len(self.weighs_per_epoch))], [i[0][0] for i in self.weighs_per_epoch])
        ax_weighs_per_epoch.plot([i for i in range(len(self.weighs_per_epoch))], [i[0][1] for i in self.weighs_per_epoch])
        ax_weighs_per_epoch.plot([i for i in range(len(self.weighs_per_epoch))], [i[0][2] for i in self.weighs_per_epoch])
        ax_weighs_per_epoch.plot([i for i in range(len(self.weighs_per_epoch))], [i[0][3] for i in self.weighs_per_epoch])
        ax_weighs_per_epoch.plot([i for i in range(len(self.weighs_per_epoch))], [i[0][4] for i in self.weighs_per_epoch])
        ax_weighs_per_epoch.set_title("Weights per epoch")
        ax_weighs_per_epoch.set_xlabel("Epoch")
        ax_weighs_per_epoch.set_ylabel("Weights")
        ax_weighs_per_epoch.legend(labels=['x0', 'x1', 'x2', 'x3', 'x4'])
        
        #save fig
        weighs_per_epoch_fig.savefig(f"./runs/{len(self.error_per_run)}/weighs.png")

        error_per_run_fig = plt.figure(figsize=(11, 10))
        ax_error_per_run = error_per_run_fig.add_subplot(111)

        labels = []
        for i in range(len(self.error_per_run)):
            ax_error_per_run.plot(
                [j for j in range(len(self.error_per_run[i]))], 
                self.error_per_run[i],
            )

            labels.append(f"Run {i + 1}, LR: {str(self.lrates[i])}, ER: {str(self.erates[i])}")
        ax_error_per_run.set_title("Error per run")
        ax_error_per_run.set_xlabel("Epoch")
        ax_error_per_run.set_ylabel("Error")
        ax_error_per_run.legend(labels)

        #save fig
        error_per_run_fig.savefig(f"./runs/{len(self.error_per_run)}/error_per_run.png")

        #save weights dataframe
        weighs_df = pd.DataFrame({
            "w0": [self.weighs_per_epoch[-1][0][0]],
            "w1": [self.weighs_per_epoch[-1][0][1]],
            "w2": [self.weighs_per_epoch[-1][0][2]],
            "w3": [self.weighs_per_epoch[-1][0][3]],
            "w4": [self.weighs_per_epoch[-1][0][4]]
        })

        data_fig = plt.figure(figsize=(9, 2))
        ax_data = data_fig.add_subplot(111)

        ax_data.spines['top'].set_visible(False)
        ax_data.spines['right'].set_visible(False)

        ax_data.xaxis.set_ticks_position('none')
        ax_data.yaxis.set_ticks_position('none')

        ax_data.axis('off')

        ax_data.table(cellText=weighs_df.values, colLabels=weighs_df.columns, loc='center')
        ax_data.set_title("Weights")

        data_fig.savefig(f"./runs/{len(self.error_per_run)}/data_w.png")
    
    def make_charts_animation(self):
        y_fig = plt.figure(figsize=(10, 10))
        self.ax_y_calc = y_fig.add_subplot(111)
        
        y_calc_animation = FuncAnimation(y_fig, self.y_caalc_per_epoch_update, frames=len(self.y_calc_per_epoch), repeat=False)
        y_calc_animation.save('y_calc.gif', writer=PillowWriter(fps=10))
    
    def y_caalc_per_epoch_update(self, frame):
        self.ax_y_calc.clear()

        self.ax_y_calc.plot([i for i in range(len(self.yd_p))], self.yd_p, label="yd")
        self.ax_y_calc.plot([i for i in range(len(self.yd_p))], self.y_calc_per_epoch[frame][0], label="y_calc")
    
    # Activation function
    def _f(self, u):
        return u

if __name__ == "__main__":
    percept = Perceptron()
    percept.mainloop()