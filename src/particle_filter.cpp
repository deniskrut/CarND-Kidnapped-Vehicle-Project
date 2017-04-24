/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "particle_filter.h"

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // Number of priticles
  num_particles = 50;
  
  // random number generator
  std::default_random_engine gen;
  
  // Normal distributions for initial coordinates
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);
  
  // Initialize the weights
  particles.resize(num_particles);
  for (int i = 0; i < num_particles; i++)
  {
    double cur_x = dist_x(gen);
    double cur_y = dist_y(gen);
    double cur_theta = dist_theta(gen);
    Particle cur_particle {/*id*/ i, /*x*/ cur_x, /*y*/ cur_y, /* theta */ cur_theta, /* weight */ 1};
    particles.push_back(cur_particle);
  }
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  // random number generator
  std::default_random_engine gen;
  
  // Update position for each particle
  for (int i = 0; i < num_particles; i++)
  {
    // Current particle
    Particle cur_particle = particles[i];
    
    // Calculate the mean for the x, y and theta
    double new_x_mean = cur_particle.x + std::abs(yaw_rate) < 0.001 ? velocity * delta_t * cos(cur_particle.theta) : velocity / yaw_rate * (sin(cur_particle.theta + yaw_rate * delta_t) - sin(cur_particle.theta));
    double new_y_mean = cur_particle.y + std::abs(yaw_rate) < 0.001 ? velocity * delta_t * sin(cur_particle.theta) : velocity / yaw_rate * (cos(cur_particle.theta) - cos(cur_particle.theta + yaw_rate * delta_t));
    double new_theta_mean = cur_particle.theta + yaw_rate * delta_t;
    
    // Normal distributions for new coordinates
    std::normal_distribution<double> dist_x(new_x_mean, std_pos[0]);
    std::normal_distribution<double> dist_y(new_y_mean, std_pos[1]);
    std::normal_distribution<double> dist_theta(new_theta_mean, std_pos[2]);
    
    // Update particle with new estimate
    cur_particle.x = dist_x(gen);
    cur_particle.y = dist_y(gen);
    cur_particle.theta = dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
  // Iterate through observations
  for (int obs_i = 0; obs_i < observations.size(); obs_i++)
  {
    // Current observation
    LandmarkObs cur_obs = observations[obs_i];
    
    // Distance to nearest prediction
    double min_distance = INFINITY;
    // Nearest prediction id
    int min_id = 0;
    
    for (int pred_i = 0; pred_i < predicted.size(); pred_i++)
    {
      // Current prediction
      LandmarkObs cur_pred = predicted[obs_i];
      
      // Distance from current observation to current prediction
      double cur_dist = dist(cur_pred.x, cur_pred.y, cur_obs.x, cur_obs.y);
      
      // If current distance is less then last known minimal distance,
      // update the minimum distance and nearest prediction id
      if (cur_dist < min_distance)
      {
        min_distance = cur_dist;
        min_id = cur_pred.id;
      }
    }
    
    // Assign nearest prediction id to currently known nearest id
    cur_obs.id = min_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
