class DPRDocument:

    '''
    Class for enforcing the shape of DPR documents.
    Everything needs a title and a body.
    '''

    title: str
    body: str

    def __init__(self, title: str, body:str):
        self.title = title
        self.body = body
