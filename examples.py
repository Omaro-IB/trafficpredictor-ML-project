"""
Bayes Network module usage examples
"""
from TrafficPredictor.Probability.bayesnet import BayesNet
from TrafficPredictor.Probability import distributions
from math import e

# # EXAMPLE 1: Rain, Time -> Traffic
# P(Rain)
rain_table = distributions.Distribution("Rain", ("heavy", "drizzle"))
rain_table.set_p("heavy", 0.1)
rain_table.set_p("drizzle", 0.9)

# P(Time)
time_table = distributions.Distribution("Time", ("early", "late"))
time_table.set_p("early", 0.5)
time_table.set_p("late", 0.5)

# P(Traffic | Rain, Time)
traffic_ctable = distributions.ConditionalDistribution(("Rain", "Time"), (("heavy", "drizzle"), ("early", "late")), "Traffic", ("light", "busy"))
traffic_ctable.set_p(("heavy", "early"), "light", 0.6)
traffic_ctable.set_p(("heavy", "early"), "busy", 0.4)
traffic_ctable.set_p(("heavy", "late"), "light", 0.1)
traffic_ctable.set_p(("heavy", "late"), "busy", 0.9)
traffic_ctable.set_p(("drizzle", "early"), "light", 0.15)
traffic_ctable.set_p(("drizzle", "early"), "busy", 0.85)
traffic_ctable.set_p(("drizzle", "late"), "light", 0.65)
traffic_ctable.set_p(("drizzle", "late"), "busy", 0.35)

# Network
bnet = BayesNet()
bnet.add_node(rain_table)
bnet.add_node(time_table)
bnet.add_node(traffic_ctable)

# What is the probability that it is drizzling, late, and traffic is busy?
p = e**bnet.probability({'Rain': 'drizzle', 'Time': 'late', 'Traffic': 'busy'})
print(p)  # 0.1575


# EXAMPLE 2: Rain -> Traffic -> Late
# P(Late)
late_table = distributions.Distribution("Late", ("y", "n"))
late_table.set_p("y", 0.25)
late_table.set_p("n", 0.75)

# P(Late | Traffic)
late_ctable = distributions.ConditionalDistribution(tuple(["Traffic"]), tuple([tuple(["light", "busy"])]), "Late", ("y", "n"))
late_ctable.set_p(tuple(["light"]), "y", 0.1)
late_ctable.set_p(tuple(["light"]), "n", 0.9)
late_ctable.set_p(tuple(["busy"]), "y", 0.3)
late_ctable.set_p(tuple(["busy"]), "n", 0.7)

# P(Traffic | Rain)
traffic_ctable2 = distributions.ConditionalDistribution(tuple(["Rain"]), tuple([tuple(["drizzle", "heavy"])]), "Traffic", ("light", "busy"))
traffic_ctable2.set_p(tuple(["drizzle"]), "light", 0.9)
traffic_ctable2.set_p(tuple(["drizzle"]), "busy", 0.1)
traffic_ctable2.set_p(tuple(["heavy"]), "light", 0.2)
traffic_ctable2.set_p(tuple(["heavy"]), "busy", 0.8)

# Network
bnet2 = BayesNet()
bnet2.add_node(rain_table)
bnet2.add_node(traffic_ctable2)
bnet2.add_node(late_ctable)

# What is the probability that it is raining heavy, there is busy traffic, and you are late?
p = e**bnet2.probability({'Rain': 'heavy', 'Traffic': 'busy', 'Late': 'y'})
print(p)  # 0.024


# EXAMPLE 3: Monty Python
guest_table = distributions.Distribution("Guest", ('door1', 'door2', 'door3'))
guest_table.set_p('door1', 1./3)
guest_table.set_p('door2', 1./3)
guest_table.set_p('door3', 1./3)

car_table = distributions.Distribution("Car", ('door1', 'door2', 'door3'))
car_table.set_p('door1', 1./3)
car_table.set_p('door2', 1./3)
car_table.set_p('door3', 1./3)

monty_table = distributions.ConditionalDistribution(("Guest", "Car"), (('door1', 'door2', 'door3'), ('door1', 'door2', 'door3')), "Monty", ('door1', 'door2', 'door3'))
monty_table.set_p(('door1', 'door1'), 'door1', 0.0)
monty_table.set_p(('door1', 'door1'), 'door2', 0.5)
monty_table.set_p(('door1', 'door1'), 'door3', 0.5)
monty_table.set_p(('door1', 'door2'), 'door1', 0.0)
monty_table.set_p(('door1', 'door2'), 'door2', 0.0)
monty_table.set_p(('door1', 'door2'), 'door3', 1.0)
monty_table.set_p(('door1', 'door3'), 'door1', 0.0)
monty_table.set_p(('door1', 'door3'), 'door2', 1.0)
monty_table.set_p(('door1', 'door3'), 'door3', 0.0)
monty_table.set_p(('door2', 'door1'), 'door1', 0.0)
monty_table.set_p(('door2', 'door1'), 'door2', 0.0)
monty_table.set_p(('door2', 'door1'), 'door3', 1.0)
monty_table.set_p(('door2', 'door2'), 'door1', 0.5)
monty_table.set_p(('door2', 'door2'), 'door2', 0.0)
monty_table.set_p(('door2', 'door2'), 'door3', 0.5)
monty_table.set_p(('door2', 'door3'), 'door1', 1.0)
monty_table.set_p(('door2', 'door3'), 'door2', 0.0)
monty_table.set_p(('door2', 'door3'), 'door3', 0.0)
monty_table.set_p(('door3', 'door1'), 'door1', 0.0)
monty_table.set_p(('door3', 'door1'), 'door2', 1.0)
monty_table.set_p(('door3', 'door1'), 'door3', 0.0)
monty_table.set_p(('door3', 'door2'), 'door1', 1.0)
monty_table.set_p(('door3', 'door2'), 'door2', 0.0)
monty_table.set_p(('door3', 'door2'), 'door3', 0.0)
monty_table.set_p(('door3', 'door3'), 'door1', 0.5)
monty_table.set_p(('door3', 'door3'), 'door2', 0.5)
monty_table.set_p(('door3', 'door3'), 'door3', 0.0)

# Network
bnet3 = BayesNet()
bnet3.add_node(guest_table)
bnet3.add_node(car_table)
bnet3.add_node(monty_table)

# What is the probability that you pick the car?
p12 = e**bnet3.probability({'Car': 'door1', 'Guest': 'door1', 'Monty': 'door2'})
p13 = e**bnet3.probability({'Car': 'door1', 'Guest': 'door1', 'Monty': 'door3'})
p21 = e**bnet3.probability({'Car': 'door2', 'Guest': 'door2', 'Monty': 'door1'})
p23 = e**bnet3.probability({'Car': 'door2', 'Guest': 'door2', 'Monty': 'door3'})
p31 = e**bnet3.probability({'Car': 'door3', 'Guest': 'door3', 'Monty': 'door1'})
p32 = e**bnet3.probability({'Car': 'door3', 'Guest': 'door3', 'Monty': 'door2'})
print(p12+p13+p21+p23+p31+p32)  # 0.33
