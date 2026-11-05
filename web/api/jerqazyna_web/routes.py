"""Тонкие роутеры: проверка параметров, вызов MapService, перевод ошибок в HTTP-коды.

Обработчики синхронные: чтение растров блокирующее, FastAPI выполняет их в пуле потоков.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from jerqazyna_web.schemas import CellResponse, HealthResponse, LayersResponse
from jerqazyna_web.service import MapService, OutsideAreaError, OutsideGridError

router = APIRouter(prefix="/api")


def get_service(request: Request) -> MapService:
    return request.app.state.service


Service = Annotated[MapService, Depends(get_service)]


@router.get("/health", response_model=HealthResponse)
def health(service: Service) -> HealthResponse:
    return HealthResponse(
        status="ok", task_id=service.task_id,
        layers_total=len(service.registry.enabled_layers),
        layers_available=service.layers_available,
    )


@router.get("/layers", response_model=LayersResponse)
def layers(service: Service) -> LayersResponse:
    return service.layers()


@router.get("/cell", response_model=CellResponse)
def cell(
    lon: Annotated[float, Query(ge=-180, le=180, description="Долгота WGS84")],
    lat: Annotated[float, Query(ge=-90, le=90, description="Широта WGS84")],
    service: Service,
) -> CellResponse:
    try:
        return service.cell(lon, lat)
    except OutsideGridError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except OutsideAreaError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
