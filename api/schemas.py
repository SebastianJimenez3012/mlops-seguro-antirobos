"""Contratos Pydantic de entrada y salida de la API."""
from typing import Optional

from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    Mto_TC: float = Field(ge=0)
    MARCA: str
    Nombre_territorio: str
    FLAG_LIMA_PROVINCIA: Optional[float] = None
    REGION: Optional[str] = None
    SUELDO_ESTIMADO: Optional[float] = Field(default=None, ge=0)
    EDAD: Optional[float] = Field(default=None, ge=18, le=100)
    SEXO: Optional[str] = None
    ANTIGUEDAD_MES: Optional[float] = Field(default=None, ge=0)
    SEGMENTO: Optional[str] = None
    FLAG_UNICEF: Optional[float] = 0


class PredictionResponse(BaseModel):
    probability: float
    prediction: int
    priority: str
