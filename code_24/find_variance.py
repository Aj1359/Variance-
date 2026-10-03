import networks as net 
import statistics

'''
This script calculates the variance of degree in Google+ and Email-EU and writes the results to a file
'''

G_google = net.get_google()

degrees = [v for (k, v) in G_google.degree()]
variance_google = statistics.variance(degrees)

G_email = net.get_emailnetwork()
degrees = [v for (k, v) in G_email.degree()]
variance_email = statistics.variance(degrees)

with open('cache/variance.txt', 'w') as out_file:
    out_file.write(f"Google {round(variance_google, 3)}\nEmail {round(variance_email, 3)}")

print(f"Google {round(variance_google, 3)}\nEmail {round(variance_email, 3)}")
