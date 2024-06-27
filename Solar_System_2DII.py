import math
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import json
import random

from dataclasses import dataclass

@dataclass
class Planet:
    position: np.ndarray
    velocity: np.ndarray
    mass: float

    def __init__(self, planet_position: list[float], planet_velocity: list[float], mass: float):
       self.position = np.array(planet_position)
       self.velocity = np.array(planet_velocity)
       self.mass = mass

def json_data(filename: str):
    data = open(filename)
    json_file = json.load(data)

    planets = []
    for value in json_file.values():
       planets.append(Planet(value['position'], value['velocity'], value['mass']))
    data.close()

    return planets

def random_planets(n):
   planets = []

   for i in range(n):
      position = [np.random.uniform(-1e13, 1e13), np.random.uniform(-1e13, 1e13)]
      velocity = [np.random.uniform(-1e7, 1e7), np.random.uniform(-1e7, 1e7)]
      mass = [np.random.uniform(1e21, 1e23)]
      planets.append(Planet(position, velocity, mass))

   return planets
    
class Solar_system:
   planets: list[Planet]
   planet_solve: int

   position2: np.ndarray
   velocity2: np.ndarray
   mass2: np.ndarray

   course: np.ndarray
   distance: np.ndarray
   gravity: np.ndarray
   acceleration: np.ndarray

   Time_position: list[np.ndarray]

   def __init__(self, planets: list[Planet]):
      self.planets = planets
      self.solve = len(planets)
      self.position2 = np.array([s.position for s in planets])
      self.velocity2 = np.array([s.velocity for s in planets])
      self.mass2 = np.array([s.mass for s in planets])
      self.course = np.zeros((self.solve, self.solve, 2))
      self.distance = np.zeros((self.solve, self.solve))
      self.gravity = np.zeros((self.solve, self.solve, 2))
      self.acceleration = np.zeros((self.solve, 2))
      self.Time_position = [np.copy(self.position2)]
   

   def solve_gravity(self):
      mass_planet = self.mass2[:, None] * self.mass2
      G = 6.6743e-11
      gravity = G * np.divide (mass_planet, np.power(self.distance, 2), out = np.zeros_like(mass_planet), where = self.distance!=0)
         
      gravity_vector = gravity[:, :, np.newaxis] * self.course
      self.gravity = gravity_vector

   def solve_distance(self):
      diff = self.position2[:, np.newaxis, :] - self.position2
      tmp = np.sum(diff * diff, axis = -1)
      self.distance = np.sqrt(tmp)
      self.course = np.divide((-diff), self.distance[:, :, np.newaxis], out = np.zeros_like(diff), where = self.distance[:, :, np.newaxis]!=[0])
   
   def solve_acceleration(self):
      full_Fg = np.sum(self.gravity, axis = (1))
      self.acceleration = np.divide(full_Fg, self.mass2[:, np.newaxis])

   def planets_moving(self, time_planet):
      self.solve_gravity()
      self.solve_distance()
      self.solve_acceleration()
      self.position2 += self.velocity2 * time_planet
      self.velocity2 += self.acceleration * time_planet
      self.Time_position.append(np.copy(self.position2))

   def progressing_simulate(self, run, time_planet):
      for i in range(math.floor(run / time_planet)):
         self.planets_moving(time_planet)

   def draw_way(self, size):
      fig, ax = plt.subplots()
      ax.set_xlim([-size, size])
      ax.set_ylim([-size, size])

      for i in range(self.solve):
          x_values = [s[i][0] for s in self.Time_position]
          y_values = [s[i][1] for s in self.Time_position]
          ax.plot(x_values, y_values)
      plt.show()
            
   def draw_place(self, size):
      fig, ax = plt.subplots()
      ax.set_xlim([-size, size])
      ax.set_ylim([-size, size])

      x_values = [s[0] for s in self.position2]
      y_values = [s[1] for s in self.position2]
      stars = ax.plot(x_values, y_values, '*')
      plt.show()

   def animation_planets(self, size):
      fig, ax = plt.subplots()
      ax.set_xlim([-size, size])
      ax.set_ylim([-size, size])
      stars = ax.plot([s[0] for s in self.Time_position[0]], [s[1] for s in self.Time_position[0]], '*')

      def animation_sol(i):
         stars.set_x([s[0] for s in self.Time_position[i]])
         stars.set_y([s[1] for s in self.Time_position[i]])
         return stars

      Solar_animation = animation.FuncAnimation(fig, func = animation_sol, frames = np.arange(0, len(self.Time_position), 1), interval = 23)
      plt.show()


