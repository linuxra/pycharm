class Duck:
    def quack(self):
        return "Quack!"

class Dog:
    def quack(self):
        return "Dog does not quack!"

def animal_sound(animal):
    print(animal.quack())

duck = Duck()
dog = Dog()

animal_sound(duck)  # Outputs: Quack!
animal_sound(dog)  # Outputs: Dog does not quack!
