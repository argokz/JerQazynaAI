# Сборка статики и раздача через nginx. Адрес API зашивается при сборке (VITE_API_URL).
FROM node:22.20.0-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY . .
ARG VITE_API_URL=http://localhost:8000
ARG VITE_BASEMAP_URL=https://tile.openstreetmap.org/{z}/{x}/{y}.png
ENV VITE_API_URL=$VITE_API_URL VITE_BASEMAP_URL=$VITE_BASEMAP_URL
RUN npm run build

FROM nginx:1.28.0-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
