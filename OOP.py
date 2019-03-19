#static method
class Employee:
    @staticmethod
    def welcomeMessage():
        print('Welcom')


"""
=======================
"""

class Apple:
    manufacturer = 'Apple Inc.'
    contactWebsite = 'www.apple.com/contact'

    def contact_details(self):
        print('To contact us, log on to', self.contactWebsite)

class MacBook(Apple):
    def __init__(self):
        self.yearOfManufacture = 2017

    def manufactureDetails(self):
        print('This MacBook was manufactured in the year {} by {}'.format(self.yearOfManufacture, self.manufacturer))

macBook = MacBook()
macBook.manufactureDetails()
macBook.contactWebsite()


