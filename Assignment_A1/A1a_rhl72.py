import numpy as np
import matplotlib.pyplot as plt

# Function to compute the normal probability density function
def g(x, mu=0, sigma=1.0):
    return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

# Function to plot the normal distribution PDF
def plot_normal_pdf(mu=0, sigma=1.0, x_range=None, point=None):
    """
    Plots the normal PDF with overlays for a given point and annotation.
    
    Parameters:
        mu (float): Mean of the normal distribution.
        sigma (float): Standard deviation of the normal distribution.
        x_range (tuple): Range for x-axis relative to the mean (default: mu ± 4sigma).
        point (float): Specific x-coordinate to annotate on the plot.
    """
    if x_range is None:
        x_range = (mu - 4 * sigma, mu + 4 * sigma)  # Default range: mean ± 4sigma
    
    # Generate x values and compute the PDF
    x_values = np.linspace(x_range[0], x_range[1], 500)
    y_values = g(x_values, mu=mu, sigma=sigma)
    
    # Create the plot
    plt.figure(figsize=(8, 5))
    plt.plot(x_values, y_values, label=f"Normal PDF (mu={mu}, sigma={sigma})")
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.title("Normal Probability Density Function")
    plt.axhline(0, color="black", linewidth=0.8, linestyle="--")  # Add x-axis
    
    # Add point and annotation if specified
    if point is not None:
        point_likelihood = g(point, mu=mu, sigma=sigma)
        plt.scatter([point], [point_likelihood], color="red", zorder=5, label=f"Point: x={point}")
        plt.plot([point, point], [0, point_likelihood], color="gray", linestyle="--", zorder=4)
        plt.annotate(
            f"{point_likelihood:.3f}",
            (point, point_likelihood),
            textcoords="offset points",
            xytext=(10, 10),
            ha="center",
            color="blue"
        )
    
    plt.legend()
    plt.grid(True)
    plt.show()


