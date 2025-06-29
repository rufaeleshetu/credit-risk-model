from pydantic import BaseModel

class InputData(BaseModel):
    age: int
    income: float
    loan_amount: float
    credit_score: int

