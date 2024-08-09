class IAAResultObj:
    def __init__(self, iaa, diff, same, unagreed):
        self.iaa = iaa
        self.diff = diff
        self.same = same
        self.unagreed = unagreed

    def __str__(self):
        return f'iaa={self.iaa}, diff={self.diff}, same={self.same}, unagreed={self.unagreed}'
