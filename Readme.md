## Информация
Бэкенд построен с использованием следующих технологий:
1. Машинное обучение - используется фреймворк Pytorch
2. API для работы с моделью - FastApi
ДатаСет был собран из изображений полученных из открытых источников вручную. Отладку модели производили с использованием Jupiter Lab. Тестировались несколько моделей. Выбрана была Pytorch, так как Она наиболее хорошо работает с большими объемами данных. В дальнейшем планируется оптимизировать модель, подобрать гиперпараметры модели. Провести кросс-валидацию. ДатаСет небольшой в планах расширить его и на его основании дообучить модель, чтобы Она могла определять ещё больше архитектурных объектов.
Модель распознаёт изображение и определяет какое это архитектурное сооружение, затем обращается через API к ГигаЧату и получает информацию про это сооружение. Затем передаёт на фронт строку в виде json.
Бэкенд написан на Python, так как это наиболее хорошо оптимизированный язык с большим количеством моделей.
Также были подготовлены контейнеры содержащие фронтенд и бэкенд проекта.
Ссылки на Них:
Фронтенд
https://hub.docker.com/r/diman4ik110/frontend-gid
Бэкенд:
