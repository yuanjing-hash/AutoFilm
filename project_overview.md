
## `/Users/yuanjing/Code/AutoFilm/requirements.txt`

```
PyYAML==6.0.1
click==8.1.7
feedparser==6.0.11
APScheduler==3.10.4
aiofile==3.8.8
httpx[http2]==0.27.2
pydantic==2.9.2
pypinyin==0.53.0
pillow==11.3.0
numpy==2.3.1
scikit-learn==1.7.0
```

## `/Users/yuanjing/Code/AutoFilm/config/config.yaml.example`

```
Settings:
  DEV: False                          # 开发者模式(可选，默认 False)

Alist2StrmList:
  - id: 动漫                          # 标识 ID
    cron: 0 20 * * *                  # 后台定时任务 Cron 表达式
    url: https://alist.akimio.top     # Alist 服务器地址
    username: admin                   # Alist 用户名
    password: adminadmin              # Alist 密码
    token: alist-d22d23ddf42fvv2      # Alist Token 永久令牌（可选，使用永久令牌则无需设置账号密码）
    source_dir: /ani/                 # Alist 服务器上文件夹路径
    target_dir: D:\media\             # 输出路径
    flatten_mode: False               # 平铺模式，开启后 subtitle、image、nfo 强制关闭(可选，默认 False)
    subtitle: False                   # 是否下载字幕文件（可选，默认 False）
    image: False                      # 是否下载图片文件（可选，默认 False）
    nfo: False                        # 是否下载 .nfo 文件（可选，默认 False）
    mode: AlistURL                    # Strm 文件中的内容（可选项：AlistURL、RawURL、AlistPath）
    overwrite: False                  # 覆盖模式，本地路径存在同名文件时是否重新生成/下载该文件（可选，默认 False）
    sync_server: True                 # 是否同步服务器（可选，默认为 True）
    sync_ignore: \.(nfo|jpg)$         # 同步时忽略的文件正则表达式（可选，默认为空，仅对文件名及拓展名有效，对路径无效）
    other_ext:                        # 自定义下载后缀，使用西文半角逗号进行分割，（可选，默认为空）
    max_workers: 50                   # 最大并发数，减轻对 Alist 服务器的负载（可选，默认 50）
    max_downloaders: 5                # 最大同时下载文件数（可选，默认 5）
    wait_time: 0                      # 遍历请求间隔时间，避免被风控，单位为秒，默认为 0

  - id: 电影
    cron: 0 0 7 * *
    url: http://alist.example2.com:5244
    username: alist
    password: alist
    token:
    source_dir: /网盘/115/电影
    target_dir: /media/my_video 
    flatten_mode: False 
    subtitle: False
    image: False
    nfo: False
    mode: RawURL
    overwrite: False
    sync_server: True
    sync_ignore:
    other_ext: .zip,.md
    max_workers: 5

Ani2AlistList:
  - id: 新番追更                           # 标识 ID
    cron: 20 12 * * *                     # 后台定时任务 Cron 表达式
    url: https://127.0.0.1:5244           # Alist 服务器地址
    username: admin                       # Alist 用户名（需管理员权限）
    password: myalist                     # Alist 密码
    token: alist-d2cac32c3c3cec2
    target_dir: /视频/动漫/新番            # Alist 地址树存储器路径，若存储器不存在将自动创建（可选，默认/Anime）
    rss_update: False                     # 使用 RSS 订阅更新最新番剧，启用后忽视传入的 year 和 month（可选，默认为 True）
    year: 2024                            # 动漫季度-年份，仅支持 2019-1 及以后更新的番剧（可选，默认使用当前日期）
    month: 7                              # 动漫季度-月份，仅支持 2019-1 及以后更新的番剧（可选，默认使用当前日期）
    src_domain: aniopen.an-i.workers.dev  # AniOpen 项目域名（可选，默认为 aniopen.an-i.workers.dev）
    rss_domain: api.ani.rip               # AniOpen 项目 RSS 订阅域名（可选，默认为 api.ani.rip） 
  
LibraryPosterList:                          # 媒体库海报更新      
  - cron: 50 13 * * *                       # 后台定时任务 Cron 表达式
    id: 我的Jellyfin                        # 任务 ID
    url: http://example.jellyfin.com:8096   # 服务器地址（支持emby和jellyfin）
    api_key: xxxxxxxxxxxxxxxx               # api key
    title_font_path: fonts/ch.ttf           # 主标题字体文件
    subtitle_font_path: fonts/en.otf        # 副标题字体文件
    configs:                                # 任务配置
      - library_name: 动漫                   # 媒体库库名（仅处理定义过的媒体库）
        title: 动漫                          # 海报大标题
        subtitle: ANIME                     # 海报副标题 
      - library_name: 国漫
        title: 国漫
        subtitle: CHINESE ANIME
      - library_name: 影视
        title: 动画电影
        subtitle: ANIME MOVIE
  - cron: 50 13 * * *                       # 可以添加多个媒体服务
    id: emby                       
    url: http://example.emby.com:8096
    api_key: xxxxxxxxxxxxxxxx
    title_font_path: fonts/ch.ttf
    subtitle_font_path: fonts/en.otf 
    configs: 
      - library_name: 日韩剧
        title: 日韩剧
        subtitle: JA & KR DRAMA
      - library_name: 欧美剧
        title: 欧美剧
        subtitle: WESTERN DRAMA
      - library_name: 综艺
        title: 综艺
        subtitle: VARIETY SHOW
```

## `/Users/yuanjing/Code/AutoFilm/app/version.py`

```
APP_VERSION = "v1.4.0"

```

## `/Users/yuanjing/Code/AutoFilm/app/utils/url.py`

```
from urllib.parse import quote, unquote, urlparse


class URLUtils:
    """
    URL 相关工具
    """

    SAFE_WORD = ";/?:@=&"

    @classmethod
    def encode(cls, url: str) -> str:
        """
        URL 编码
        """
        return quote(url, safe=cls.SAFE_WORD)

    @staticmethod
    def decode(strings: str) -> str:
        """
        URL 解码
        """
        return unquote(strings)

    @staticmethod
    def get_resolve_url(url: str) -> tuple[str, str, int]:
        """
        从 URL 中解析协议、域名和端口号

        未知端口号的情况下，端口号设为 -1
        """
        parsed_result = urlparse(url)

        scheme = parsed_result.scheme
        netloc = parsed_result.netloc

        # 去除用户信息
        if "@" in netloc:
            netloc = netloc.split("@")[-1]

        # 处理域名和端口
        if ":" in netloc:
            domain, port_str = netloc.rsplit(":", 1)
            try:
                port = int(port_str)
            except ValueError:
                port = -1  # 端口号解析失败，设为 0
        else:
            domain = netloc
            if scheme == "http":
                port = 80
            elif scheme == "https":
                port = 443
            else:
                port = -1  # 未知协议，端口号设为 0

        return scheme, domain, port

```

## `/Users/yuanjing/Code/AutoFilm/app/utils/strings.py`

```
from pypinyin import pinyin, Style


class StringsUtils:
    """
    字符串工具类
    """

    @staticmethod
    def get_pinyin(text: str) -> str:
        """
        获取中文字符串的拼音
        :param text: 中文字符串
        :return: 拼音字符串
        """
        return "".join([item[0] for item in pinyin(text, style=Style.NORMAL)])

    @staticmethod
    def get_initials(text: str) -> str:
        """
        获取中文字符串的首字母
        :param text: 中文字符串
        :return: 首字母字符串
        """
        return "".join([item[0] for item in pinyin(text, style=Style.FIRST_LETTER)])

    @staticmethod
    def get_cn_ascii(text: str) -> str:
        """
        获取中文字符串的 ASCCII 字符串
        :param text: 中文字符串
        :return: ASCCII 字符串
        """
        return "".join(hex(ord(char))[2:] for char in text)  # 移除 hex 前缀 0x

```

## `/Users/yuanjing/Code/AutoFilm/app/utils/singleton.py`

```
import abc


class Singleton(abc.ABCMeta, type):
    """
    单例模式
    """

    _instances: dict = {}

    def __call__(cls, *args, **kwargs):
        key = cls
        if key not in cls._instances:
            cls._instances[key] = super().__call__(*args, **kwargs)
        return cls._instances[key]


if __name__ == "__main__":
    # 示例单例类
    class MySingleton1(metaclass=Singleton):
        def __init__(self, value):
            self.value = value

    class MySingleton2(metaclass=Singleton):
        def __init__(self, value):
            self.value = value

    # 测试单例
    instance1 = MySingleton1(10)
    instance2 = MySingleton1(20)
    intance3 = MySingleton2(10)

    print(instance1 is instance2)  # 输出: True
    print(instance1 is intance3)  # 输出: False
    print(instance1.value)  # 输出: 10
    print(instance2.value)  # 输出: 10
    print(intance3.value)  # 输出: 10

```

## `/Users/yuanjing/Code/AutoFilm/app/utils/retry.py`

```
from asyncio import sleep as async_sleep
from typing import Type, Callable, ParamSpec, TypeVar, Optional, Awaitable
from time import sleep
from functools import wraps

from app.core.log import logger
from app.utils.singleton import Singleton

P = ParamSpec("P")
R = TypeVar("R")


class Retry(metaclass=Singleton):
    """
    重试装饰器
    """

    TRIES: int = 3  # 默认最大重试次数
    DELAY: int = 3  # 默认延迟时间
    BACKOFF: int = 1  # 默认延迟倍数

    WARNING_MSG: str = "{}，{}秒后重试 ..."
    ERROR_MSG: str = "{}，超出最大重试次数！"

    @classmethod
    def sync_retry(
        cls,
        exception: Type[Exception],
        tries: int = TRIES,
        delay: int = DELAY,
        backoff: int = BACKOFF,
    ) -> Callable[[Callable[P, R]], Callable[P, Optional[R]]]:
        """
        同步重试装饰器

        :param exception: 需要捕获的异常
        :param tries: 最大重试次数
        :param delay: 延迟时间
        :param backoff: 延迟倍数
        :param ret: 默认返回
        """

        def inner(func: Callable[P, R]) -> Callable[P, Optional[R]]:
            @wraps(func)
            def wrapper(*args, **kwargs) -> Optional[R]:
                remaining_retries = tries
                while remaining_retries > 0:
                    try:
                        return func(*args, **kwargs)
                    except exception as e:
                        remaining_retries -= 1
                        if remaining_retries >= 0:
                            _delay = (tries - remaining_retries) * backoff * delay
                            logger.warning(cls.WARNING_MSG.format(e, _delay))
                            sleep(_delay)
                        else:
                            logger.error(cls.ERROR_MSG.format(e))
                            return None

            return wrapper

        return inner

    @classmethod
    def async_retry(
        cls,
        exception: Type[Exception],
        tries: int = TRIES,
        delay: int = DELAY,
        backoff: int = BACKOFF,
    ) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[Optional[R]]]]:
        """
        异步重试装饰器

        :param exception: 需要捕获的异常
        :param tries: 最大重试次数
        :param delay: 延迟时间
        :param backoff: 延迟倍数
        """

        def inner(
            func: Callable[P, Awaitable[R]],
        ) -> Callable[P, Awaitable[Optional[R]]]:
            @wraps(func)
            async def wrapper(*args, **kwargs) -> Optional[R]:
                remaining_retries = tries
                while remaining_retries > 0:
                    try:
                        return await func(*args, **kwargs)
                    except exception as e:
                        remaining_retries -= 1
                        if remaining_retries >= 0:
                            _delay = (tries - remaining_retries) * backoff * delay
                            logger.warning(cls.WARNING_MSG.format(e, _delay))
                            await async_sleep(_delay)
                        else:
                            logger.error(cls.ERROR_MSG.format(e))
                            return None

            return wrapper

        return inner

```

## `/Users/yuanjing/Code/AutoFilm/app/utils/photo.py`

```
from io import BytesIO
from base64 import b64encode

import numpy as np
from PIL import Image, ImageFilter, ImageDraw, ImageFont
from sklearn.cluster import KMeans  # type: ignore


class PhotoUtils:
    @staticmethod
    def get_primary_color(
        img: Image.Image, num_colors: int = 5, bg_clusters: int = 1
    ) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        """
        从PIL图像对象中提取主题色（背景色）和适配的文字颜色
        :param img: PIL图像对象
        :param num_colors: 要提取的主色数量（默认5）
        :param bg_clusters: 用于合并背景色的主色数量（默认1）
        :return: 返回一个元组，包含背景色和文字颜色
        格式为 ((r, g, b), (r, g, b))
        其中背景色是RGB格式，文字颜色是根据背景色亮度计算的
        """
        # 将PIL图像转换为NumPy数组 (RGB格式)
        img_array = np.array(img)

        # 如果图像有透明通道，移除alpha通道
        if img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]

        # 将图像数据重塑为2D数组 (像素 x RGB)
        pixel_data = img_array.reshape((-1, 3))

        # 使用K-Means聚类提取主色
        kmeans = KMeans(n_clusters=num_colors, n_init=10, random_state=42)
        kmeans.fit(pixel_data)

        # 获取聚类中心和对应的像素数量
        colors = kmeans.cluster_centers_
        counts = np.bincount(kmeans.labels_)

        # 按出现频率排序颜色
        sorted_indices = np.argsort(counts)[::-1]
        main_colors = colors[sorted_indices].astype(int)

        # 合并指定数量的聚类作为背景色
        background_color = np.mean(main_colors[:bg_clusters], axis=0).astype(int)

        # 计算背景色的相对亮度
        r, g, b = background_color
        luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255

        # 根据亮度选择文字颜色
        text_color = (0, 0, 0) if luminance > 0.5 else (255, 255, 255)

        return tuple(background_color), tuple(text_color)

    @staticmethod
    def create_gradient_background(
        width: int,
        height: int,
        color: tuple[int, int, int],
    ) -> Image.Image:
        """
        创建一个从左到右的渐变背景，使用遮罩技术实现渐变效果
        左侧颜色更深，右侧颜色适中，提供更明显的渐变效果
        :param width: 背景宽度
        :param height: 背景高度
        :param color: 主题色，格式为 (r, g, b)
        :return: 渐变背景图像
        """
        base = Image.new("RGB", (width, height), color)  # 创建基础图像（右侧原始颜色）

        # 创建渐变遮罩（水平方向：左黑右白）
        gradient = Image.new("L", (width, 1))  # 单行渐变
        gradient_data = []
        for x in range(width):
            # 计算渐变值：左侧0（全黑），右侧255（全白）
            value = int(255 * x / max(1, width - 1))
            gradient_data.append(value)

        # 应用渐变数据并垂直拉伸
        gradient.putdata(gradient_data)
        mask = gradient.resize((width, height))

        # 创建暗色版本（左侧颜色）
        dark_factor = 0.5  # 颜色加深系数
        dark_color = (
            int(color[0] * dark_factor),
            int(color[1] * dark_factor),
            int(color[2] * dark_factor),
        )
        dark = Image.new("RGB", (width, height), dark_color)

        # 使用遮罩混合两种颜色
        return Image.composite(base, dark, mask)

    @staticmethod
    def add_shadow(
        img: Image.Image, offset=(5, 5), shadow_color=(0, 0, 0, 100), blur_radius=3
    ) -> Image.Image:
        """
        给图片添加右侧和底部阴影
        :param img: 原始图片（PIL.Image对象）
        :param offset: 阴影偏移量，(x, y)格式
        :param shadow_color: 阴影颜色，RGBA格式
        :param blur_radius: 阴影模糊半径
        :return: 添加了阴影的新图片
        """
        # 创建一个透明背景，比原图大一些，以容纳阴影
        shadow_width = img.width + offset[0] + blur_radius * 2
        shadow_height = img.height + offset[1] + blur_radius * 2

        shadow = Image.new("RGBA", (shadow_width, shadow_height), (0, 0, 0, 0))
        shadow_layer = Image.new("RGBA", img.size, shadow_color)  # 创建阴影层

        # 将阴影层粘贴到偏移位置
        shadow.paste(shadow_layer, (blur_radius + offset[0], blur_radius + offset[1]))
        # 模糊阴影
        shadow = shadow.filter(ImageFilter.GaussianBlur(blur_radius))
        # 创建结果图像
        result = Image.new("RGBA", shadow.size, (0, 0, 0, 0))
        # 将原图粘贴到结果图像上
        result.paste(
            img, (blur_radius, blur_radius), img if img.mode == "RGBA" else None
        )
        # 合并阴影和原图（保持原图在上层）
        return Image.alpha_composite(shadow, result)

    @staticmethod
    def apply_rounded_corners(image: Image.Image, radius: int) -> Image.Image:
        """应用圆角效果"""
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.rounded_rectangle([(0, 0), image.size], radius=radius, fill=255)
        result = Image.new("RGBA", image.size, (0, 0, 0, 0))
        result.paste(image, (0, 0), mask)
        return result

    @staticmethod
    def draw_text_on_image(
        image: Image.Image,
        text: str,
        position: tuple[int, int],
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
        fill_color: tuple[int, int, int] = (255, 255, 255),
        shadow_enabled: bool = False,
        shadow_color: tuple[int, int, int, int] = (0, 0, 0, 180),
        shadow_offset: tuple[int, int] = (2, 2),
    ) -> None:
        """
        在图像上绘制文字，可选添加文字阴影
        :param image: PIL.Image对象
        :param text: 要绘制的文字
        :param position: 文字位置 (x, y)
        :param font: 字体对象
        :param font_size: 字体大小
        :param fill_color: 文字颜色，RGB格式
        :param shadow_enabled: 是否启用文字阴影
        :param shadow_color: 阴影颜色，RGBA格式
        :param shadow_offset: 阴影偏移量，(x, y)格式
        :return: 添加了文字的图像
        """
        draw = ImageDraw.Draw(image)
        # 如果启用阴影，先绘制阴影文字
        if shadow_enabled:
            shadow_position = (
                position[0] + shadow_offset[0],
                position[1] + shadow_offset[1],
            )
            draw.text(shadow_position, text, font=font, fill=shadow_color)
        # 绘制正常文字
        draw.text(position, text, font=font, fill=fill_color)

    @staticmethod
    def draw_multiline_text_on_image(
        image: Image.Image,
        texts: list[str],
        position: tuple[int, int],
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
        font_size: int,
        line_spacing: int = 10,
        fill_color=(255, 255, 255),
        shadow_enabled=False,
        shadow_color=(0, 0, 0, 180),
        shadow_offset=(2, 2),
    ):
        """
        在图像上绘制多行文字，根据空格自动换行，可选添加文字阴影

        :param image: PIL.Image对象
        :param text: 要绘制的文字
        :param position: 第一行文字位置 (x, y)
        :param font: 字体对象
        :param font_size: 字体大小
        :param line_spacing: 行间距
        :param fill_color: 文字颜色，RGBA格式
        :param shadow_enabled: 是否启用文字阴影
        :param shadow_color: 阴影颜色，RGB格式
        :param shadow_offset: 阴影偏移量，(x, y)格式
        """
        draw = ImageDraw.Draw(image)
        x, y = position
        for i, line in enumerate(texts):
            current_y = y + i * (font_size + line_spacing)
            if shadow_enabled:  # 如果启用阴影，先绘制阴影文字
                shadow_x = x + shadow_offset[0]
                shadow_y = current_y + shadow_offset[1]
                draw.text((shadow_x, shadow_y), line, font=font, fill=shadow_color)
            draw.text((x, current_y), line, font=font, fill=fill_color)  # 绘制正常文字

    @staticmethod
    def encode_image(image: Image.Image, format: str = "PNG") -> bytes:
        """
        将PIL图像编码为base64字节数据（用于API上传）
        :param image: PIL图像对象
        :param format: 图像格式，默认为PNG
        :return: 图像的base64编码字节数据
        """
        buffer = BytesIO()
        if image.mode in ("RGBA", "LA"):  # 确保图像是RGB格式（移除透明通道）
            background = Image.new("RGB", image.size, (255, 255, 255))  # 创建白色背景
            if image.mode == "RGBA":
                background.paste(image, mask=image.split()[-1])  # 使用alpha通道作为mask
            else:
                background.paste(image)
            image = background
        elif image.mode != "RGB":
            image = image.convert("RGB")

        image.save(buffer, format=format)
        buffer.seek(0)
        return b64encode(buffer.getvalue())  # 返回base64编码的字节数据

```

## `/Users/yuanjing/Code/AutoFilm/app/utils/multiton.py`

```
import abc


class Multiton(abc.ABCMeta, type):
    """
    多例模式
    """

    _instances: dict = {}

    def __call__(cls, *args, **kwargs):
        key = (cls, args, frozenset(kwargs.items()))
        if key not in cls._instances:
            cls._instances[key] = super().__call__(*args, **kwargs)
        return cls._instances[key]


if __name__ == "__main__":
    # 示例多例类
    class MyMultiton1(metaclass=Multiton):
        def __init__(self, value):
            self.value = value

    class MyMultiton2(metaclass=Multiton):
        def __init__(self, value):
            self.value = value

    # 测试多例
    instance1 = MyMultiton1(10)
    instance2 = MyMultiton1(20)
    instance3 = MyMultiton1(10)
    instance4 = MyMultiton2(10)

    print(instance1 is instance2)  # 输出: False
    print(instance1 is instance3)  # 输出: True
    print(instance1 is instance4)  # 输出: False
    print(instance1.value)  # 输出: 10
    print(instance2.value)  # 输出: 20
    print(instance3.value)  # 输出: 10
    print(instance4.value)  # 输出: 10

```

## `/Users/yuanjing/Code/AutoFilm/app/utils/http.py`

```
from typing import Any, Literal, overload
from pathlib import Path
from os import makedirs
from asyncio import TaskGroup, to_thread
from collections.abc import Coroutine
from tempfile import TemporaryDirectory
from shutil import copy
from weakref import WeakSet

from httpx import AsyncClient, Client, Response, TimeoutException
from aiofile import async_open

from app.core import settings, logger
from app.utils.url import URLUtils
from app.utils.retry import Retry


class HTTPClient:
    """
    HTTP 客户端类
    """

    # 最小流式下载文件大小，128MB
    MINI_STREAM_SIZE: int = 128 * 1024 * 1024
    # 默认请求头
    HEADERS: dict[str, str] = {
        "User-Agent": f"AutoFilm/{settings.APP_VERSION}",
        "Accept": "application/json",
    }

    def __init__(self):
        """
        初始化 HTTP 客户端
        """

        self.__new_async_client()
        self.__new_sync_client()

    def __new_sync_client(self):
        """
        创建新的同步 HTTP 客户端
        """
        self.__sync_client = Client(http2=True, follow_redirects=True, timeout=10)

    def __new_async_client(self):
        """
        创建新的异步 HTTP 客户端
        """
        self.__async_client = AsyncClient(http2=True, follow_redirects=True, timeout=10)

    def close_sync_client(self) -> None:
        """
        关闭同步 HTTP 客户端
        """
        if self.__sync_client:
            self.__sync_client.close()

    async def close_async_client(self) -> None:
        """
        关闭异步 HTTP 客户端
        """
        if self.__async_client:
            await self.__async_client.aclose()

    @Retry.sync_retry(TimeoutException, tries=3, delay=1, backoff=2)
    def _sync_request(self, method: str, url: str, **kwargs) -> Response | None:
        """
        发起同步 HTTP 请求
        """
        try:
            return self.__sync_client.request(method, url, **kwargs)
        except TimeoutException as e:
            self.close_sync_client()
            self.__new_sync_client()
            raise TimeoutException(f"HTTP 请求超时：{e}")

    @Retry.async_retry(TimeoutException, tries=3, delay=1, backoff=2)
    async def _async_request(self, method: str, url: str, **kwargs) -> Response | None:
        """
        发起异步 HTTP 请求
        """
        try:
            return await self.__async_client.request(method, url, **kwargs)
        except TimeoutException as e:
            await self.close_async_client()
            self.__new_async_client()
            raise TimeoutException(f"HTTP 请求超时：{e}")

    @overload
    def request(
        self, method: str, url: str, *, sync: Literal[True], **kwargs
    ) -> Response | None: ...

    @overload
    def request(
        self, method: str, url: str, *, sync: Literal[False] = False, **kwargs
    ) -> Coroutine[Any, Any, Response | None]: ...

    def request(
        self,
        method: str,
        url: str,
        *,
        sync: Literal[True, False] = False,
        **kwargs,
    ) -> Response | None | Coroutine[Any, Any, Response | None]:
        """
        发起 HTTP 请求

        :param method: HTTP 方法，如 get, post, put 等
        :param url: 请求的 URL
        :param sync: 是否使用同步请求方式，默认为 False
        :param kwargs: 其他请求参数，如 headers, cookies 等
        :return: HTTP 响应对象
        """
        headers = kwargs.get("headers", self.HEADERS)
        kwargs["headers"] = headers
        if sync:
            return self._sync_request(method, url, **kwargs)
        else:
            return self._async_request(method, url, **kwargs)

    @overload
    def head(self, url: str, *, sync: Literal[True], **kwargs) -> Response | None: ...

    @overload
    def head(
        self, url: str, *, sync: Literal[False], **kwargs
    ) -> Coroutine[Any, Any, Response | None]: ...

    def head(
        self,
        url: str,
        *,
        sync: Literal[True, False] = False,
        params: dict = {},
        **kwargs,
    ) -> Response | None | Coroutine[Any, Any, Response | None]:
        """
        发送 HEAD 请求

        :param url: 请求的 URL
        :param sync: 是否使用同步请求方式，默认为 False
        :param params: 请求的查询参数
        :param kwargs: 其他请求参数，如 headers, cookies 等
        :return: HTTP 响应对象
        """
        return self.request("head", url, sync=sync, params=params, **kwargs)

    @overload
    def get(self, url: str, *, sync: Literal[True], **kwargs) -> Response | None: ...

    @overload
    def get(
        self, url: str, *, sync: Literal[False], **kwargs
    ) -> Coroutine[Any, Any, Response | None]: ...

    def get(
        self,
        url: str,
        *,
        sync: Literal[True, False] = False,
        params: dict = {},
        **kwargs,
    ) -> Response | None | Coroutine[Any, Any, Response | None]:
        """
        发送 GET 请求

        :param url: 请求的 URL
        :param sync: 是否使用同步请求方式，默认为 False
        :param params: 请求的查询参数
        :param kwargs: 其他请求参数，如 headers, cookies 等
        :return: HTTP 响应对象
        """
        return self.request("get", url, sync=sync, params=params, **kwargs)

    @overload
    def post(self, url: str, *, sync: Literal[True], **kwargs) -> Response | None: ...

    @overload
    def post(
        self, url: str, *, sync: Literal[False], **kwargs
    ) -> Coroutine[Any, Any, Response] | None: ...

    def post(
        self,
        url: str,
        *,
        sync: Literal[True, False] = False,
        data: Any = None,
        json: dict = {},
        **kwargs,
    ) -> Response | None | Coroutine[Any, Any, Response | None]:
        """
        发送 POST 请求

        :param url: 请求的 URL
        :param sync: 是否使用同步请求方式，默认为 False
        :param data: 请求的数据
        :param json: 请求的 JSON 数据
        :param kwargs: 其他请求参数，如 headers, cookies 等
        :return: HTTP 响应对象
        """
        return self.request("post", url, sync=sync, data=data, json=json, **kwargs)

    @overload
    def put(self, url: str, *, sync: Literal[True], **kwargs) -> Response | None: ...

    @overload
    def put(
        self, url: str, *, sync: Literal[False], **kwargs
    ) -> Coroutine[Any, Any, Response | None]: ...

    def put(
        self,
        url: str,
        *,
        sync: Literal[True, False] = False,
        data: Any = None,
        json: dict = {},
        **kwargs,
    ) -> Response | None | Coroutine[Any, Any, Response | None]:
        """
        发送 PUT 请求

        :param url: 请求的 URL
        :param sync: 是否使用同步请求方式，默认为 False
        :param data: 请求的数据
        :param json: 请求的 JSON 数据
        :param kwargs: 其他请求参数，如 headers, cookies 等
        :return: HTTP 响应对象
        """
        return self.request("put", url, sync=sync, data=data, json=json, **kwargs)

    async def download(
        self,
        url: str,
        file_path: Path,
        params: dict = {},
        chunk_num: int = 5,
        **kwargs,
    ) -> None:
        """
        下载文件！！！仅支持异步下载！！！

        :param url: 文件的 URL
        :param file_path: 文件保存路径
        :param params: 请求参数
        :param kwargs: 其他请求参数，如 headers, cookies 等
        """
        resp = await self.head(url, sync=False, params=params, **kwargs)

        file_size = int(resp.headers.get("Content-Length", -1))

        with TemporaryDirectory(prefix="AutoFilm_") as temp_dir:  # 创建临时目录
            temp_file = Path(temp_dir) / file_path.name

            if file_size == -1:
                logger.debug(f"{file_path.name} 文件大小未知，直接下载")
                await self.__download_chunk(url, temp_file, 0, 0, **kwargs)
            else:
                async with TaskGroup() as tg:
                    logger.debug(
                        f"开始分片下载文件：{file_path.name}，分片数:{chunk_num}"
                    )
                    for start, end in self.caculate_divisional_range(
                        file_size, chunk_num=chunk_num
                    ):
                        tg.create_task(
                            self.__download_chunk(url, temp_file, start, end, **kwargs)
                        )
            copy(temp_file, file_path)

    async def __download_chunk(
        self,
        url: str,
        file_path: Path,
        start: int,
        end: int,
        iter_chunked_size: int = 64 * 1024,
        **kwargs,
    ):
        """
        下载文件的分片

        :param url: 文件的 URL
        :param file_path: 文件保存路径
        :param start: 分片的开始位置
        :param end: 分片的结束位置
        :param iter_chunked_size: 下载的块大小（下载完成后再写入硬盘），默认为 64KB
        :param kwargs: 其他请求参数，如 headers, cookies, proxies 等
        """

        await to_thread(makedirs, file_path.parent, exist_ok=True)

        if start != 0 and end != 0:
            headers = kwargs.get("headers", {})
            headers["Range"] = f"bytes={start}-{end}"
            kwargs["headers"] = headers

        resp = await self.get(url, sync=False, **kwargs)
        async with async_open(file_path, "ab") as file:
            file.seek(start)
            async for chunk in resp.aiter_bytes(iter_chunked_size):
                await file.write(chunk)

    @staticmethod
    def caculate_divisional_range(
        file_size: int,
        chunk_num: int,
    ) -> list[tuple[int, int]]:
        """
        计算文件的分片范围

        :param file_size: 文件大小
        :param chunk_num: 分片数
        :return: 分片范围
        """
        if file_size < HTTPClient.MINI_STREAM_SIZE or chunk_num <= 1:
            return [(0, file_size - 1)]

        step = file_size // chunk_num  # 计算每个分片的基本大小
        remainder = file_size % chunk_num  # 计算剩余的字节数

        chunks = []
        start = 0

        for i in range(chunk_num):
            # 如果有剩余字节，分配一个给当前分片
            end = start + step + (1 if i < remainder else 0) - 1
            chunks.append((start, end))
            start = end + 1

        return chunks


class RequestUtils:
    """
    HTTP 请求工具类
    支持同步和异步请求
    """

    __clients: dict[str, HTTPClient] = {}
    __client_list: WeakSet[HTTPClient] = WeakSet()

    @classmethod
    def get_client(cls, url: str = "") -> HTTPClient:
        """
        获取 HTTP 客户端

        :param url: 请求的 URL
        :return: HTTP 客户端
        """

        if url:
            _, domain, port = URLUtils.get_resolve_url(url)
            key = f"{domain}:{port}"
            if key not in cls.__clients:
                cls.__clients[key] = HTTPClient()
            return cls.__clients[key]

        client = HTTPClient()
        cls.__client_list.add(client)
        return client

    @overload
    @classmethod
    def request(
        cls, method: str, url: str, sync: Literal[True], **kwargs
    ) -> Response | None: ...

    @overload
    @classmethod
    def request(
        cls, method: str, url: str, sync: Literal[False] = False, **kwargs
    ) -> Coroutine[Any, Any, Response | None]: ...

    @classmethod
    def request(
        cls, method: str, url: str, sync: Literal[True, False] = False, **kwargs
    ) -> Response | None | Coroutine[Any, Any, Response | None]:
        """
        发起 HTTP 请求
        """
        client = cls.get_client(url)
        return client.request(method, url, sync=sync, **kwargs)

    @overload
    @classmethod
    def head(cls, url: str, sync: Literal[True], **kwargs) -> Response | None: ...

    @overload
    @classmethod
    def head(
        cls, url: str, sync: Literal[False] = False, **kwargs
    ) -> Coroutine[Any, Any, Response | None]: ...

    @classmethod
    def head(
        cls,
        url: str,
        *,
        sync: Literal[True, False] = False,
        params: dict = {},
        **kwargs,
    ) -> Response | None | Coroutine[Any, Any, Response | None]:
        """
        发送 HEAD 请求

        :param url: 请求的 URL
        :param sync: 是否使用同步请求方式，默认为 False
        :param params: 请求的查询参数
        :param kwargs: 其他请求参数，如 headers, cookies 等
        :return: HTTP 响应对象
        """
        return cls.request("head", url, sync=sync, params=params, **kwargs)

    @overload
    @classmethod
    def get(cls, url: str, *, sync: Literal[True], **kwargs) -> Response | None: ...

    @overload
    @classmethod
    def get(
        cls, url: str, *, sync: Literal[False] = False, **kwargs
    ) -> Coroutine[Any, Any, Response | None]: ...

    @classmethod
    def get(
        cls,
        url: str,
        *,
        sync: Literal[True, False] = False,
        params: dict = {},
        **kwargs,
    ) -> Response | None | Coroutine[Any, Any, Response | None]:
        """
        发送 GET 请求

        :param url: 请求的 URL
        :param params: 请求的查询参数
        :param kwargs: 其他请求参数，如 headers, cookies 等
        :return: HTTP 响应对象
        """
        return cls.request("get", url, sync=sync, params=params, **kwargs)

    @overload
    @classmethod
    def post(cls, url: str, *, sync: Literal[True], **kwargs) -> Response | None: ...

    @overload
    @classmethod
    def post(
        cls,
        url: str,
        *,
        sync: Literal[False] = False,
        data: Any = None,
        json: dict = {},
        **kwargs,
    ) -> Coroutine[Any, Any, Response | None]: ...

    @classmethod
    def post(
        cls,
        url: str,
        *,
        sync: Literal[True, False] = False,
        data: Any = None,
        json: dict = {},
        **kwargs,
    ) -> Response | None | Coroutine[Any, Any, Response | None]:
        """
        发送 POST 请求

        :param url: 请求的 URL
        :param data: 请求的数据
        :param json: 请求的 JSON 数据
        :param kwargs: 其他请求参数，如 headers, cookies 等
        :return: HTTP 响应对象
        """
        return cls.request("post", url, sync=sync, data=data, json=json, **kwargs)

    @overload
    @classmethod
    def put(cls, url: str, *, sync: Literal[True], **kwargs) -> Response | None: ...

    @overload
    @classmethod
    def put(
        cls,
        url: str,
        *,
        sync: Literal[False] = False,
        data: Any = None,
        **kwargs,
    ) -> Coroutine[Any, Any, Response | None]: ...

    @classmethod
    def put(
        cls, url: str, *, sync: Literal[True, False] = False, data: Any = None, **kwargs
    ) -> Response | None | Coroutine[Any, Any, Response | None]:
        """
        发送 PUT 请求

        :param key: 客户端的键
        :param url: 请求的 URL
        :param data: 请求的数据
        :param kwargs: 其他请求参数，如 headers, cookies 等
        :return: HTTP 响应对象
        """
        return cls.request("put", url, sync=sync, data=data, **kwargs)

    @classmethod
    async def download(
        cls,
        url: str,
        file_path: Path,
        params: dict = {},
        **kwargs,
    ) -> None:
        """
        下载文件！！！仅支持异步下载！！！

        :param url: 文件的 URL
        :param file_path: 文件保存路径
        :param params: 请求参数
        :param kwargs: 其他请求参数，如 headers, cookies 等
        """
        client = cls.get_client(url)
        await client.download(url, file_path, params=params, **kwargs)

```

## `/Users/yuanjing/Code/AutoFilm/app/utils/alist.py`

```
from hmac import new as hmac_new
from hashlib import sha256 as hashlib_sha256
from base64 import urlsafe_b64encode

from app.utils.singleton import Singleton


class AlistUtils(metaclass=Singleton):
    """
    Alist 相关工具
    """

    @staticmethod
    def sign(secret_key: str, data: str) -> str:
        """
        计算 Alist 签名
        :param secret_key: Alist 签名 Token
        :param data: Alist 文件绝对路径（未编码）
        """

        if not secret_key:
            return ""
        else:
            h = hmac_new(secret_key.encode(), digestmod=hashlib_sha256)
            expire_time_stamp = str(0)
            h.update((data + ":" + expire_time_stamp).encode())
            return f"?sign={urlsafe_b64encode(h.digest()).decode()}:0"

    @staticmethod
    def structure2dict(text: str) -> dict:
        """
        将能够被 Alist 地址树识别的文本转换为字典，支持键值对中包含两个冒号的情况
        """
        lines = text.strip().split("\n")
        current_folder: str = ""

        def parse_lines(
            start_index: int = 0, indent_level: int = 0
        ) -> tuple[dict, int]:
            result_dict = {}
            i = start_index
            while i < len(lines):
                line = lines[i]
                current_indent = len(line) - len(line.lstrip())

                if current_indent > indent_level:
                    sub_dict, new_index = parse_lines(i, current_indent)
                    result_dict[current_folder] = sub_dict
                    i = new_index
                    continue

                elif current_indent < indent_level:
                    break

                else:
                    parts = line.strip().split(":")
                    if len(parts) == 5:
                        key, value1, value2, value3 = (
                            parts[0].strip(),
                            parts[1].strip(),
                            parts[2].strip(),
                            ":".join(parts[3:]).strip(),
                        )
                        result_dict[key] = [value1, value2, value3]
                    elif len(parts) == 4:
                        key, value1, value2 = (
                            parts[0].strip(),
                            parts[1].strip(),
                            ":".join(parts[2:]).strip(),
                        )
                        result_dict[key] = [value1, value2]
                    elif len(parts) >= 3:
                        key, value = parts[0].strip(), ":".join(parts[1:]).strip()
                        result_dict[key] = value
                    else:
                        current_folder = parts[0]
                        result_dict[current_folder] = {}
                    i += 1

            return result_dict, i

        result_dict, _ = parse_lines()
        return result_dict

    @staticmethod
    def dict2structure(dictionary: dict) -> str:
        """
        将字典转换为能够被 Alist 地址树识别的文本
        """

        def parse_dict(
            sub_dictionary: dict[str, str | list[str] | dict], indent: int = 0
        ):
            result_str = ""
            for key, value in sub_dictionary.items():
                if isinstance(value, str):
                    result_str += " " * indent + f"{key}:{value}\n"
                elif isinstance(value, list):
                    result_str += " " * indent + f"{key}:{':'.join(value)}\n"
                elif isinstance(value, dict):
                    result_str += " " * indent + f"{key}:\n"
                    result_str += parse_dict(value, indent + 2)

                if indent == 0 and result_str.startswith(":"):
                    result_str = result_str.lstrip(":").strip()

            return result_str

        return parse_dict(dictionary)

```

## `/Users/yuanjing/Code/AutoFilm/app/utils/__init__.py`

```
from app.utils.http import RequestUtils, HTTPClient
from app.utils.alist import AlistUtils
from app.utils.retry import Retry
from app.utils.url import URLUtils
from app.utils.singleton import Singleton
from app.utils.multiton import Multiton
from app.utils.strings import StringsUtils
from app.utils.photo import PhotoUtils

__all__ = [
    RequestUtils,
    HTTPClient,
    AlistUtils,
    Retry,
    URLUtils,
    Singleton,
    Multiton,
    StringsUtils,
    PhotoUtils,
]

```

## `/Users/yuanjing/Code/AutoFilm/app/modules/themoviedb.py`

```
import requests
import logging
from typing import Optional


class TheMovieDateBase:
    """
    调用 TMDB  官方 APIv3 获取影视作品信息
    官方 API 文档：https://developers.themoviedb.org/3/
    """

    def __init__(
        self, api_key: str, domain: str = "api.themoviedb.org", language: str = "zh-CN"
    ) -> None:
        """
        实例化 TheMovieDateBase 对象

        :param api_key: TMDB API Key(v3)
        :param domain: TMDB API 域名，默认 "api.themoviedb.org"
        :param language: 语言，默认 "zh-CN"
        """

        self.api_key = api_key
        self.api_url = f"https://{domain}/3"
        self.language = language

        self.timeout = 5

    def search(
        self,
        query_keyword: str,
        page: int = 1,
        media_type: Optional[str] = "multi",
    ) -> Optional[dict]:
        """
        根据关键字匹配剧集，获取相关信息

        :param query_keyword: 查询关键字
        :param page: 查询页数，默认 1
        :param media_type: 查询类型，可选 "multi", "movie", "tv"，默认 "multi"
        :return: 返回查询结果
        """

        if media_type not in ("multi", "movie", "tv"):
            logging.error(f"media_type 参数错误，仅支持 multi, movie, tv 三种类型！")
            return

        url = f"{self.api_url}/search/{media_type}"
        params = {
            "api_key": self.api_key,
            "language": self.language,
            "query": query_keyword,
            "page": page,
        }

        return requests.get(url=url, params=params).json()

    def movie_details(self, movie_id: int) -> Optional[dict]:
        """
        根据 movie_id 查询详细电影信息

        :param movie_id: 电影 ID
        :return: 返回查询结果
        """

        url = f"{self.api_url}/movie/{movie_id}"
        params = {
            "api_key": self.api_key,
            "language": self.language,
            "movie_id": movie_id,
        }

        return requests.get(url=url, params=params).json()

    def tv_details(self, tv_id: int, season: int = 1) -> Optional[dict]:
        """
        根据 tv_id 查询详细电视剧信息

        :param tv_id: 电视剧 ID
        :param season: 季数，默认 1
        :return: 返回查询结果
        """

        url = f"{self.api_url}/tv/{tv_id}/season/{season}"
        params = {
            "api_key": self.api_key,
            "language": self.language,
        }

        return requests.get(url=url, params=params).json()

```

## `/Users/yuanjing/Code/AutoFilm/app/modules/libraryposter/poster.py`

```
import random
from pathlib import Path
from io import BytesIO
from typing import Any, AsyncGenerator

from PIL import Image, ImageDraw, ImageFont

from app.core import logger
from app.utils import RequestUtils, PhotoUtils


class LibraryPoster:
    def __init__(
        self,
        url: str,
        api_key: str,
        title_font_path: str,
        subtitle_font_path: str,
        configs: list[dict[str, str]],
        **_,
    ) -> None:
        """
        初始化库海报更新客户端
        :param url: 服务器地址
        :param api_key: API 密钥
        :param title_font_path: 主标题字体路径
        :param subtitle_font_path: 副标题字体路径
        """
        self.__server_url = url
        self.__api_key = api_key
        self.__title_font_path = Path(title_font_path)
        self.__subtitle_font_path = Path(subtitle_font_path)
        self.__configs = configs

    async def get_users(self) -> list[dict[str, Any]]:
        """
        获取用户列表
        :return: 用户列表
        """
        resp = await RequestUtils.get(
            f"{self.__server_url}/Users?api_key={self.__api_key}"
        )
        if resp is None:
            logger.warning(f"获取 {self.__server_url} 用户列表失败")
            return []

        if resp.status_code != 200:
            logger.warning(
                f"获取 {self.__server_url} 用户列表失败, 状态码: {resp.status_code}"
            )
            return []
        return resp.json()

    async def get_libraries(self) -> list[dict[str, Any]]:
        """
        返回媒体库列表
        """
        resp = await RequestUtils.get(
            f"{self.__server_url}/Library/MediaFolders?api_key={self.__api_key}",
        )
        if resp is None:
            logger.warning(f"获取 {self.__server_url} 媒体库列表失败")
            return []

        if resp.status_code != 200:
            logger.warning(
                f"获取 {self.__server_url} 媒体库列表失败, 状态码: {resp.status_code}"
            )
            return []

        return resp.json()["Items"]

    async def get_library_items(
        self,
        library_id: str,
        user_id: str = "",
    ) -> list[dict[str, Any]]:
        """
        获取指定媒体库的详细信息
        :param library_id: 媒体库 ID
        :param user_id: 用户 ID（可选）
        :return: 媒体库项目列表
        """
        if not user_id:
            users = await self.get_users()
            if not users:
                logger.warning("未找到任何用户，无法获取媒体库项目")
                return []
            user_id = users[0]["Id"]  # 默认使用第一个用户

        url = f"{self.__server_url}/Users/{user_id}/Items?ParentId={library_id}&api_key={self.__api_key}"
        resp = await RequestUtils.get(url)

        if resp is None or resp.status_code != 200:
            logger.warning(
                f"获取 {library_id} 媒体库信息失败, 状态码: {resp.status_code if resp else '无响应'}"
            )
            return []

        return resp.json()["Items"]

    async def download_item_image(
        self,
        item: dict[str, Any],
        image_type: str = "Primary",
    ) -> Image.Image | None:
        """
        下载指定项目海报图片
        :param item: 项目字典
        :return: 图片字节内容
        """
        url = f"{self.__server_url}/Items/{item['Id']}/Images/{image_type}?api_key={self.__api_key}"
        resp = await RequestUtils.get(url)

        if resp is None or resp.status_code != 200:
            logger.warning(
                f"下载项目 {item['Name']} {image_type} 类型图片失败, 状态码: {resp.status_code if resp else '无响应'}"
            )
            return None

        return Image.open(BytesIO(resp.content))

    async def download_library_poster(
        self,
        library: dict[str, Any],
    ) -> AsyncGenerator[Image.Image, None]:
        """
        下载媒体库海报
        :param library: 媒体库字典
        :return:
        """
        logger.info(f"开始下载 {library['Name']} 媒体库的海报图片")
        for items in await self.get_library_items(library["Id"]):
            image = await self.download_item_image(items)
            if image is not None:
                yield image

    def process_poster(
        self,
        images: list[Image.Image],
        title: str = "",
        subtitle: str = "",
        width: int = 1920,
        height: int = 1080,
    ) -> Image.Image:
        """
        处理海报图片，将图片布局在右半边
        :param images: 图片列表
        :param title: 海报标题
        :param subtitle: 海报副标题
        :param width: 背景宽度
        :param height: 背景高度
        :return: 处理后的海报图片
        """

        logger.info(f"开始处理海报图片，标题: {title}, 副标题: {subtitle}")
        # 随机打乱图片顺序
        random.shuffle(images)

        # 布局参数
        COLS = 3
        ROWS = 3

        # 动态计算图片尺寸（相对于背景尺寸）
        CELL_WIDTH = int(width * 0.20)  # 约占背景宽度的20%
        CELL_HEIGHT = int(CELL_WIDTH * 1.5)  # 保持海报比例 2:3

        # 动态计算间距
        COLUMN_SPACING = int(width * 0.025)  # 列间距约占2.5%
        ROW_SPACING = int(height * 0.05)  # 行间距约占5%

        CORNER_RADIUS = max(15, int(CELL_WIDTH * 0.08))  # 圆角半径
        ROTATION_ANGLE = -18  # 旋转角度
        RIGHT_MARGIN = int(width * 0.05)  # 右边距占5%

        # 根据图片大小自适应计算阴影参数
        # 文字阴影偏移：基于字体大小和背景尺寸
        text_shadow_offset_x = max(2, int(width * 0.002))  # 最小2px，约占宽度的0.2%
        text_shadow_offset_y = max(2, int(height * 0.003))  # 最小2px，约占高度的0.3%
        text_shadow_offset = (text_shadow_offset_x, text_shadow_offset_y)

        # 图片阴影参数：基于图片尺寸
        img_shadow_offset_x = max(3, int(CELL_WIDTH * 0.015))
        img_shadow_offset_y = max(3, int(CELL_HEIGHT * 0.012))
        img_shadow_offset = (img_shadow_offset_x, img_shadow_offset_y)
        img_shadow_blur = max(
            2, int(min(CELL_WIDTH, CELL_HEIGHT) * 0.012)
        )  # 模糊半径基于图片较小边

        # 获取主题色并创建背景
        theme_color, text_color = PhotoUtils.get_primary_color(random.choice(images))
        background = PhotoUtils.create_gradient_background(width, height, theme_color)

        draw = ImageDraw.Draw(background)

        title_font_size = int(height * 0.15)
        subtitle_font_size = int(height * 0.06)

        try:
            title_font = ImageFont.truetype(
                self.__title_font_path.as_posix(), size=title_font_size
            )
            subtitle_font = ImageFont.truetype(
                self.__subtitle_font_path.as_posix(), size=subtitle_font_size
            )
        except Exception as e:
            logger.warning(f"加载自定义字体失败: {e}")
            logger.warning(
                f"主标题字体路径: {self.__title_font_path} (存在: {self.__title_font_path.exists()})"
            )
            logger.warning(
                f"副标题字体路径: {self.__subtitle_font_path} (存在: {self.__subtitle_font_path.exists()})"
            )
            title_font = ImageFont.load_default()
            subtitle_font = ImageFont.load_default()

        # 主标题在左侧中间偏上
        title_bbox = draw.textbbox((0, 0), title, font=title_font)
        title_text_w = title_bbox[2] - title_bbox[0]
        left_half_center = width // 4  # 左半边的中心位置
        title_x = left_half_center - title_text_w // 2
        title_y = int(height * 0.35)  # 中间偏上

        PhotoUtils.draw_text_on_image(
            background,
            title,
            (title_x, title_y),
            title_font,
            text_color,
            shadow_enabled=False,
            shadow_offset=text_shadow_offset,
        )

        # 副标题在左侧中间偏下
        subtitle_bbox = draw.textbbox((0, 0), subtitle, font=subtitle_font)
        subtitle_text_w = subtitle_bbox[2] - subtitle_bbox[0]
        subtitle_x = left_half_center - subtitle_text_w // 2
        subtitle_y = int(height * 0.60)  # 中间偏下

        PhotoUtils.draw_text_on_image(
            background,
            subtitle,
            (subtitle_x, subtitle_y),
            subtitle_font,
            text_color,
            shadow_enabled=True,
            shadow_offset=text_shadow_offset,
        )

        # 计算右半边区域
        right_half_start = width // 2
        available_width = width // 2 - RIGHT_MARGIN

        # 计算网格尺寸（允许超出右半边）
        grid_width = COLS * CELL_WIDTH + (COLS - 1) * COLUMN_SPACING
        grid_height = ROWS * CELL_HEIGHT + (ROWS - 1) * ROW_SPACING

        # 网格起始位置 - 让中心图片在右半边居中，周围可以超出
        grid_center_x = right_half_start + int(available_width * 0.7)
        grid_center_y = height // 2

        grid_start_x = grid_center_x - grid_width // 2
        grid_start_y = grid_center_y - grid_height // 2

        # 处理每个海报
        processed_count = 0
        for col in range(COLS):
            for row in range(ROWS):
                if processed_count >= len(images):
                    break

                img = images[processed_count]
                processed_count += 1

                # 1. 调整图片大小
                resized = img.resize((CELL_WIDTH, CELL_HEIGHT), Image.LANCZOS)

                # 2. 应用圆角
                rounded = PhotoUtils.apply_rounded_corners(resized, CORNER_RADIUS)

                # 3. 添加阴影
                shadowed = PhotoUtils.add_shadow(
                    rounded, offset=img_shadow_offset, blur_radius=img_shadow_blur
                )

                # 4. 计算基础位置 - 让每列的中心点在一条斜线上
                # 原始网格中心位置
                original_center_x = (
                    grid_start_x + col * (CELL_WIDTH + COLUMN_SPACING) + CELL_WIDTH // 2
                )
                original_center_y = (
                    grid_start_y + row * (CELL_HEIGHT + ROW_SPACING) + CELL_HEIGHT // 2
                )

                # 根据旋转角度计算水平偏移，使斜线效果在旋转后仍然保持
                vertical_offset = height * col * 0.03
                target_center_y = original_center_y + vertical_offset

                # 5. 应用旋转
                rotated = shadowed.rotate(
                    ROTATION_ANGLE,
                    expand=True,
                    fillcolor=(0, 0, 0, 0),
                    resample=Image.BICUBIC,
                )

                # 6. 计算旋转后的最终位置，确保旋转中心对齐
                pos_x = original_center_x - rotated.width / 2
                pos_y = target_center_y - rotated.height / 2

                # 7. 粘贴到背景
                background.paste(rotated, (int(pos_x), int(pos_y)), rotated)

        logger.info(f"成功处理 {processed_count} 张海报图片")
        return background

    async def update_library_image(
        self, library: dict[str, Any], image: Image.Image, image_type: str = "Primary"
    ) -> None:
        """
        更新媒体库的海报图片
        :param library: 媒体库字典
        :param image: 要更新的图片
        :param image_type: 图片类型，默认为 Primary
        """
        url = f"{self.__server_url}/Items/{library['Id']}/Images/{image_type}?api_key={self.__api_key}"
        headers = {
            "Content-Type": "image/png",
        }

        image_data_base64 = PhotoUtils.encode_image(image=image, format="PNG")
        resp = await RequestUtils.post(url, data=image_data_base64, headers=headers)
        if resp is None or resp.status_code != 204:
            logger.warning(
                f"更新 {library['Name']} 媒体库图片失败, 状态码: {resp.status_code if resp else '无响应'}"
            )
        else:
            logger.info(f"成功更新 {library['Name']} 媒体库 {image_type} 类型的图片")

    async def process_library(
        self,
        library: dict[str, Any],
        title: str = "",
        subtitle: str = "",
        limit: int = 15,
    ) -> None:
        """
        处理单个媒体库
        :param library: 媒体库项目字典
        :param title: 海报标题
        :param subtitle: 海报副标题
        :param limit: 限制下载的图片数量
        """

        images: list[Image.Image] = []
        async for image in self.download_library_poster(library):
            images.append(image)
            if len(images) >= limit:
                break

        logger.info(f"获取到 {library['Name']} 媒体库的 {len(images)} 张海报图片")
        result = self.process_poster(images, title, subtitle)
        await self.update_library_image(library, result)
        logger.info(f"媒体库 {library['Name']} 的海报图片处理成功")

    async def run(self) -> None:
        """
        执行库海报更新
        """
        libraries = await self.get_libraries()
        library_kv: dict[str, str] = {item["Name"]: item for item in libraries}
        for config in self.__configs:
            if config["library_name"] in library_kv:
                await self.process_library(
                    library_kv[config["library_name"]],
                    config.get("title", ""),
                    config.get("subtitle", ""),
                )

```

## `/Users/yuanjing/Code/AutoFilm/app/modules/libraryposter/__init__.py`

```
from app.modules.libraryposter.poster import LibraryPoster

__all__ = [
    LibraryPoster,
]

```

## `/Users/yuanjing/Code/AutoFilm/app/modules/filetransfer/recognize.py`

```
from re import compile, findall, I

from app.extensions import RELEASEGROUP


def match_relasegroup(title: str = None) -> str:
    """
    匹配资源发布/字幕/制作组

    :param title: 资源标题或文件名
    :return: 匹配结果
    """
    if not title:
        return ""

    release_groups = "|".join(RELEASEGROUP)
    title = title + " "
    groups_re = compile(r"(?<=[-@\[￡【&])(?:%s)(?=[@.\s\]\[】&])" % release_groups, I)
    recognized_groups = []
    for recognized_group in findall(groups_re, title):
        if recognized_group not in recognized_groups:
            recognized_groups.append(recognized_group)
    return "@".join(recognized_groups)

```

## `/Users/yuanjing/Code/AutoFilm/app/modules/filetransfer/__init__.py`

```
"""
文件整理/刮削模块
"""
```

## `/Users/yuanjing/Code/AutoFilm/app/modules/ani2alist/ani2alist.py`

```
from typing import Final
from datetime import datetime

from feedparser import parse  # type:ignore

from app.core import logger
from app.utils import RequestUtils, URLUtils
from app.utils import AlistUtils
from app.modules.alist import AlistClient

VIDEO_MINETYPE: Final = frozenset(("video/mp4", "video/x-matroska"))
SUBTITLE_MINETYPE: Final = frozenset(("application/octet-stream",))
ZIP_MINETYPE: Final = frozenset(("application/zip",))
FILE_MINETYPE: Final = VIDEO_MINETYPE | SUBTITLE_MINETYPE | ZIP_MINETYPE

ANI_SEASION: Final = frozenset((1, 4, 7, 10))


class Ani2Alist:
    """
    将 ANI Open 项目的视频通过地址树的方式挂载在 Alist服务器上
    """

    def __init__(
        self,
        url: str = "http://localhost:5244",
        username: str = "",
        password: str = "",
        token: str = "",
        target_dir: str = "/Anime",
        rss_update: bool = True,
        year: int | None = None,
        month: int | None = None,
        src_domain: str = "aniopen.an-i.workers.dev",
        rss_domain: str = "api.ani.rip",
        key_word: str | None = None,
        **_,
    ) -> None:
        """
        实例化 Ani2Alist 对象

        :param origin: Alist 服务器地址，默认为 "http://localhost:5244"
        :param username: Alist 用户名，默认为空
        :param password: Alist 密码，默认为空
        :param token: Alist Token，默认为空
        :param target_dir: 挂载到 Alist 服务器上目录，默认为 "/Anime"
        :param rss_update: 使用 RSS 追更最新番剧，默认为 True
        :param year: 动画年份，默认为空
        :param month: 动画季度，默认为空
        :param src_domain: ANI Open 项目地址，默认为 "aniopen.an-i.workers.dev"，可自行反代
        :param rss_domain ANI Open 项目 RSS 地址，默认为 "api.ani.rip"，可自行反代
        :param key_word: 自定义关键字，默认为空
        """

        self.client = AlistClient(url, username, password, token)
        self.__target_dir = "/" + target_dir.strip("/")

        self.__year: int | None = None
        self.__month: int | None = None
        self.__key_word: str | None = None
        self.__rss_update: bool = rss_update

        if rss_update:
            logger.debug("使用 RSS 追更最新番剧")
        elif key_word:
            logger.debug(f"使用自定义关键字：{key_word}")
            self.__key_word = key_word
        elif year and month:
            self.__year = year
            self.__month = month
        elif year or month:
            logger.warning("未传入完整时间参数，默认使用当前季度")
        else:
            logger.info("未传入时间参数，默认使用当前季度")

        self.__src_domain = src_domain.strip()
        self.__rss_domain = rss_domain.strip()

    async def run(self) -> None:
        is_valid, error_msg = self.__is_valid()
        if not is_valid:
            logger.error(error_msg)
            return

        storage = await self.client.get_storage_by_mount_path(
            mount_path=self.__target_dir,
            create=True,
            driver="UrlTree",
        )
        if storage is None:
            logger.error(f"未找到挂载路径：{self.__target_dir}，并且无法创建")
            return

        addition_dict = storage.addition2dict
        url_dict = AlistUtils.structure2dict(addition_dict.get("url_structure", ""))

        await self.__update_url_dicts(url_dict)

        addition_dict["url_structure"] = AlistUtils.dict2structure(url_dict)
        storage.set_addition_by_dict(addition_dict)

        await self.client.async_api_admin_storage_update(storage)

    async def __update_url_dicts(self, url_dict: dict):
        """
        更新 URL 字典
        """
        if self.__rss_update:
            await self.update_rss_anime_dict(url_dict)
        else:
            await self.update_season_anime_dict(url_dict)

    def __is_valid(self) -> tuple[bool, str]:
        """
        判断参数是否合理
        :return: (是否合理, 错误信息)
        """
        if self.__rss_update:
            return True, ""
        if self.__year is None and self.__month is None:
            return True, ""
        current_date = datetime.now()
        if (self.__year, self.__month) == (2019, 4):
            return False, "2019-4季度暂无数据"
        elif (self.__year, self.__month) < (2019, 1):
            return False, "ANI Open 项目仅支持2019年1月及其之后的数据"
        elif (self.__year, self.__month) > (current_date.year, current_date.month):
            return False, "传入的年月晚于当前时间"
        else:
            return True, ""

    async def update_season_anime_dict(self, url_dict: dict):
        """
        更新指定季度/关键字的动画列表
        """

        def get_key() -> str:
            """
            根据 self.__year 和 self.__month 以及关键字 self.__key_word 返回关键字
            """
            if self.__key_word:
                return self.__key_word

            if self.__year and self.__month:
                year = self.__year
                month = self.__month
            else:
                current_date = datetime.now()
                year = current_date.year
                month = current_date.month

            for _month in range(month, 0, -1):
                if _month in ANI_SEASION:
                    return f"{year}-{_month}"

        def __parse2timestamp(time_str: str) -> int:
            """
            将 RSS 订阅中时间字符串转换为时间戳
            """
            dt = datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S.%fZ")
            return int(dt.timestamp())

        async def update_data(_url: str, _url_dict: dict):
            """
            用于递归更新解析数据
            """
            logger.debug(f"请求地址：{_url}")
            _resp = await RequestUtils.post(_url)
            if _resp.status_code != 200:
                raise Exception(f"请求发送失败，状态码：{_resp.status_code}")

            _result = _resp.json()

            for file in _result["files"]:
                mimeType: str = file["mimeType"]
                name: str = file["name"]
                quoted_name = URLUtils.encode(name)

                if mimeType in FILE_MINETYPE:
                    size: str = file["size"]
                    created_time_stamp: str = str(
                        __parse2timestamp(file["createdTime"])
                    )
                    __url = _url + quoted_name + "?d=true"
                    logger.debug(
                        f"获取文件：{name}，文件大小：{int(size) / 1024 / 1024:.2f}MB，播放地址：{__url}"
                    )
                    _url_dict[name] = [
                        size,
                        created_time_stamp,
                        __url,
                    ]
                elif mimeType == "application/vnd.google-apps.folder":
                    logger.debug(f"获取目录：{name}")
                    if name not in _url_dict:
                        _url_dict[name] = {}
                    await update_data(_url + quoted_name + "/", _url_dict[name])
                else:
                    logger.warning(f"无法识别类型：{mimeType}，文件详情：{file}")

        key = get_key()
        if key not in url_dict:
            url_dict[key] = {}
        await update_data(f"https://{self.__src_domain}/{key}/", url_dict[key])
        return

    async def update_rss_anime_dict(self, url_dict: dict):
        """
        更新 RSS 动画列表
        """

        def __parse2timestamp(time_str: str) -> int:
            """
            将 RSS 订阅中时间字符串转换为时间戳
            """
            dt = datetime.strptime(time_str, "%a, %d %b %Y %H:%M:%S %Z")
            return int(dt.timestamp())

        def handle_recursive(url_dict: dict, entry) -> None:
            """
            处理 RSS 数据，解析 URL 多级目录
            """
            parents = URLUtils.decode(entry.link).split("/")[3:]  # 拆分多级目录
            current_dict = url_dict
            for index in range(len(parents)):
                name = parents[index]
                if index == len(parents) - 1:
                    current_dict[entry.title] = [
                        str(convert_size_to_bytes(entry.anime_size)),
                        str(__parse2timestamp(entry.published)),
                        entry.link,
                    ]
                else:
                    if name not in current_dict:
                        current_dict[name] = {}
                    current_dict = current_dict[name]

        def convert_size_to_bytes(size_str: str) -> int:
            """
            将带单位的大小转换为字节
            """
            units = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}
            number, unit = [string.strip() for string in size_str.split()]
            return int(float(number) * units[unit])

        resp = await RequestUtils.get(f"https://{self.__rss_domain}/ani-download.xml")
        if resp.status_code != 200:
            raise Exception(f"请求发送失败，状态码：{resp.status_code}")
        feeds = parse(resp.text)

        for entry in feeds.entries:
            """
            print(type(entry))
            print(entry)

            type: <class 'feedparser.util.FeedParserDict'>
            {
                "title": "[ANi] FAIRY TAIL 魔導少年 百年任務 - 18 [1080P][Baha][WEB-DL][AAC AVC][CHT].mp4",
                "title_detail": {
                    "type": "text/plain",
                    "language": None,
                    "base": "",
                    "value": "[ANi] FAIRY TAIL 魔導少年 百年任務 - 18 [1080P][Baha][WEB-DL][AAC AVC][CHT].mp4",
                },
                "links": [
                    {
                        "rel": "alternate",
                        "type": "text/html",
                        "href": "https://resources.ani.rip/2024-7/%5BANi%5D%20FAIRY%20TAIL%20%E9%AD%94%E5%B0%8E%E5%B0%91%E5%B9%B4%20%E7%99%BE%E5%B9%B4%E4%BB%BB%E5%8B%99%20-%2018%20%5B1080P%5D%5BBaha%5D%5BWEB-DL%5D%5BAAC%20AVC%5D%5BCHT%5D.mp4?d=true",
                    }
                ],
                "link": "https://resources.ani.rip/2024-7/%5BANi%5D%20FAIRY%20TAIL%20%E9%AD%94%E5%B0%8E%E5%B0%91%E5%B9%B4%20%E7%99%BE%E5%B9%B4%E4%BB%BB%E5%8B%99%20-%2018%20%5B1080P%5D%5BBaha%5D%5BWEB-DL%5D%5BAAC%20AVC%5D%5BCHT%5D.mp4?d=true",
                "id": "https://resources.ani.rip/2024-7/%5BANi%5D%20FAIRY%20TAIL%20%E9%AD%94%E5%B0%8E%E5%B0%91%E5%B9%B4%20%E7%99%BE%E5%B9%B4%E4%BB%BB%E5%8B%99%20-%2018%20%5B1080P%5D%5BBaha%5D%5BWEB-DL%5D%5BAAC%20AVC%5D%5BCHT%5D.mp4?d=true",
                "guidislink": False,
                "published": "Sun, 10 Nov 2024 09:01:47 GMT",
                "published_parsed": time.struct_time(
                    tm_year=2024,
                    tm_mon=11,
                    tm_mday=10,
                    tm_hour=9,
                    tm_min=1,
                    tm_sec=47,
                    tm_wday=6,
                    tm_yday=315,
                    tm_isdst=0,
                ),
                "anime_size": "473.0 MB",
            }
            """
            handle_recursive(url_dict, entry)

```

## `/Users/yuanjing/Code/AutoFilm/app/modules/ani2alist/__init__.py`

```
"""
将 ANI Open 的视频通过文件树的方式挂载到 Alist上
"""
from app.modules.ani2alist.ani2alist import Ani2Alist
```

## `/Users/yuanjing/Code/AutoFilm/app/modules/alist2strm/mode.py`

```
from enum import Enum

class Alist2StrmMode(Enum):
    """
    模块 alist2strm 的运行模式
    """
    AlistURL = "AlistURL"
    RawURL = "RawURL"
    AlistPath = "AlistPath"

    @classmethod
    def from_str(cls, mode_str: str) -> "Alist2StrmMode":
        """
        从字符串转换为 AList2StrmMode 枚举
        如果字符串不匹配任何枚举值，则返回 AlistURL 模式
        :param mode_str: 模式字符串
        :return: Alist2StrmMode 枚举值
        例如，"alisturl" 将返回 Alist2StrmMode.AlistURL
        """
        return cls[mode_str.upper()] if mode_str.upper() in cls.__members__ else cls.AlistURL
```

## `/Users/yuanjing/Code/AutoFilm/app/modules/alist2strm/alist2strm.py`

```
from asyncio import to_thread, Semaphore, TaskGroup
from os import PathLike
from pathlib import Path
from re import compile as re_compile
import traceback

from aiofile import async_open

from app.core import logger
from app.utils import RequestUtils
from app.extensions import VIDEO_EXTS, SUBTITLE_EXTS, IMAGE_EXTS, NFO_EXTS
from app.modules.alist import AlistClient, AlistPath
from app.modules.alist2strm.mode import Alist2StrmMode

class Alist2Strm:
    def __init__(
        self,
        url: str = "http://localhost:5244",
        username: str = "",
        password: str = "",
        token: str = "",
        source_dir: str = "/",
        target_dir: str | PathLike = "",
        flatten_mode: bool = False,
        subtitle: bool = False,
        image: bool = False,
        nfo: bool = False,
        mode: str = "AlistURL",
        overwrite: bool = False,
        other_ext: str = "",
        max_workers: int = 50,
        max_downloaders: int = 5,
        wait_time: float | int = 0,
        sync_server: bool = False,
        sync_ignore: str | None = None,
        **_,
    ) -> None:
        """
        实例化 Alist2Strm 对象

        :param url: Alist 服务器地址，默认为 "http://localhost:5244"
        :param username: Alist 用户名，默认为空
        :param password: Alist 密码，默认为空
        :param source_dir: 需要同步的 Alist 的目录，默认为 "/"
        :param target_dir: strm 文件输出目录，默认为当前工作目录
        :param flatten_mode: 平铺模式，将所有 Strm 文件保存至同一级目录，默认为 False
        :param subtitle: 是否下载字幕文件，默认为 False
        :param image: 是否下载图片文件，默认为 False
        :param nfo: 是否下载 .nfo 文件，默认为 False
        :param mode: Strm模式(AlistURL/RawURL/AlistPath)
        :param overwrite: 本地路径存在同名文件时是否重新生成/下载该文件，默认为 False
        :param sync_server: 是否同步服务器，启用后若服务器中删除了文件，也会将本地文件删除，默认为 True
        :param other_ext: 自定义下载后缀，使用西文半角逗号进行分割，默认为空
        :param max_workers: 最大并发数
        :param max_downloaders: 最大同时下载
        :param wait_time: 遍历请求间隔时间，单位为秒，默认为 0
        :param sync_ignore: 同步时忽略的文件正则表达式
        """

        self.client = AlistClient(url, username, password, token)
        self.mode = Alist2StrmMode.from_str(mode)

        self.source_dir = source_dir
        self.target_dir = Path(target_dir)

        self.flatten_mode = flatten_mode
        if flatten_mode:
            subtitle = image = nfo = False

        download_exts: set[str] = set()
        if subtitle:
            download_exts |= SUBTITLE_EXTS
        if image:
            download_exts |= IMAGE_EXTS
        if nfo:
            download_exts |= NFO_EXTS
        if other_ext:
            download_exts |= frozenset(other_ext.lower().split(","))

        self.download_exts = download_exts
        self.process_file_exts = VIDEO_EXTS | download_exts

        self.overwrite = overwrite
        self.__max_workers = Semaphore(max_workers)
        self.__max_downloaders = Semaphore(max_downloaders)
        self.wait_time = wait_time
        self.sync_server = sync_server

        if sync_ignore:
            self.sync_ignore_pattern = re_compile(sync_ignore)
        else:
            self.sync_ignore_pattern = None

    async def run(self) -> None:
        """
        处理主体
        """
        
        # BDMV 处理相关变量初始化
        self.bdmv_collections: dict[str, list[tuple[AlistPath, int]]] = {}  # BDMV目录 -> [(文件路径, 文件大小)]
        self.bdmv_largest_files: dict[str, AlistPath] = {}  # BDMV目录 -> 最大文件路径

        def filter(path: AlistPath) -> bool:
            """
            过滤器
            根据 Alist2Strm 配置判断是否需要处理该文件
            将云盘上上的文件对应的本地文件路径保存至 self.processed_local_paths

            :param path: AlistPath 对象
            """

            if path.is_dir:
                return False

            # 跳过系统文件夹和不需要的文件
            if any(folder in path.full_path for folder in ["@eaDir", "Thumbs.db", ".DS_Store"]):
                return False

            # 完全跳过 BDMV 文件夹内的所有文件（除了我们特殊处理的 .m2ts 文件）
            if "/BDMV/" in path.full_path and not self._is_bdmv_file(path):
                logger.debug(f"跳过 BDMV 文件夹内的文件: {path.name}")
                return False

            if path.suffix.lower() not in self.process_file_exts:
                logger.debug(f"文件 {path.name} 不在处理列表中")
                return False

            # 检查是否为 BDMV 文件
            if self._is_bdmv_file(path):
                self._collect_bdmv_file(path)
                # 暂时不处理，等收集完所有文件后再决定
                return False

            try:
                local_path = self.__get_local_path(path)
            except OSError as e:  # 可能是文件名过长
                logger.warning(f"获取 {path.full_path} 本地路径失败：{e}")
                return False

            self.processed_local_paths.add(local_path)

            if not self.overwrite and local_path.exists():
                if path.suffix in self.download_exts:
                    local_path_stat = local_path.stat()
                    if local_path_stat.st_mtime < path.modified_timestamp:
                        logger.debug(
                            f"文件 {local_path.name} 已过期，需要重新处理 {path.full_path}"
                        )
                        return True
                    if local_path_stat.st_size < path.size:
                        logger.debug(
                            f"文件 {local_path.name} 大小不一致，可能是本地文件损坏，需要重新处理 {path.full_path}"
                        )
                        return True
                logger.debug(
                    f"文件 {local_path.name} 已存在，跳过处理 {path.full_path}"
                )
                return False

            return True


        if self.mode == Alist2StrmMode.RawURL:
            is_detail = True
        else:
            is_detail = False

        self.processed_local_paths = set()  # 云盘文件对应的本地文件路径

        # 第一阶段：收集所有文件信息并直接处理普通文件
        async with self.__max_workers, TaskGroup() as tg:
            async for path in self.client.iter_path(
                dir_path=self.source_dir,
                wait_time=self.wait_time,
                is_detail=is_detail,
                filter=filter,
            ):
                # 直接处理普通文件，不需要额外的 list
                tg.create_task(self.__file_processer(path))

        # 完成 BDMV 文件收集，确定最大文件
        self._finalize_bdmv_collections()
        
        # 第二阶段：处理 BDMV 最大文件
        logger.info(f"开始处理 {len(self.bdmv_largest_files)} 个 BDMV 目录")
        for bdmv_root, largest_file in self.bdmv_largest_files.items():
            try:
                logger.info(f"处理 BDMV 目录: {bdmv_root}")
                logger.info(f"最大文件: {largest_file.full_path}")
                
                # 重新获取详细信息以确保有 raw_url
                if self.mode == Alist2StrmMode.RawURL and not largest_file.raw_url:
                    logger.debug(f"重新获取 BDMV 文件详细信息: {largest_file.full_path}")
                    try:
                        updated_path = await self.client.async_api_fs_get(largest_file.full_path)
                        # 保持原有的 full_path，只更新其他属性
                        original_full_path = largest_file.full_path
                        largest_file = updated_path
                        largest_file.full_path = original_full_path
                    except Exception as e:
                        logger.warning(f"重新获取 BDMV 文件详细信息失败: {e}")
                
                # 处理文件
                await self.__file_processer(largest_file)
                
                # 添加到已处理路径列表
                local_path = self.__get_local_path(largest_file)
                self.processed_local_paths.add(local_path)
                
                logger.info(f"BDMV 文件处理完成: {largest_file.name}")
            except Exception as e:
                logger.error(f"处理 BDMV 文件 {largest_file.full_path} 时出错：{e}")
                logger.error(f"详细错误信息: {traceback.format_exc()}")
                continue

        if self.sync_server:
            await self.__cleanup_local_files()
            logger.info("清理过期的 .strm 文件完成")
        logger.info("Alist2Strm 处理完成")

    async def __file_processer(self, path: AlistPath) -> None:
        """
        异步保存文件至本地

        :param path: AlistPath 对象
        """
        local_path = self.__get_local_path(path)
        logger.debug(f"__file_processer: 处理文件 {path.full_path} -> 本地路径 {local_path} | 模式 {self.mode}")

        # 统一的 URL 生成逻辑，BDMV 文件与普通文件使用相同的逻辑
        if self.mode == Alist2StrmMode.AlistURL:
            content = path.download_url
        elif self.mode == Alist2StrmMode.RawURL:
            content = path.raw_url
        elif self.mode == Alist2StrmMode.AlistPath:
            content = path.full_path

        logger.debug(f"__file_processer: 初始 content = {content}")

        if not content:
            logger.warning(f"文件 {path.full_path} 的内容为空，跳过处理")
            return

        await to_thread(local_path.parent.mkdir, parents=True, exist_ok=True)

        logger.debug(f"开始处理 {local_path} | 内容: {content}")
        if local_path.suffix == ".strm":
            async with async_open(local_path, mode="w", encoding="utf-8") as file:
                await file.write(content)
            logger.info(f"{local_path.name} 创建成功")
        else:
            async with self.__max_downloaders:
                await RequestUtils.download(path.download_url, local_path)
                logger.info(f"{local_path.name} 下载成功")

    def __get_local_path(self, path: AlistPath) -> Path:
        """
        根据给定的 AlistPath 对象和当前的配置，计算出本地文件路径。

        :param path: AlistPath 对象
        :return: 本地文件路径
        """
        # 检查是否为 BDMV 文件
        if self._is_bdmv_file(path):
            bdmv_root = self._get_bdmv_root_dir(path)
            if bdmv_root and self._should_process_bdmv_file(path):
                # 为 BDMV 文件生成特殊路径
                movie_title = self._get_movie_title_from_bdmv_path(bdmv_root)
                
                if self.flatten_mode:
                    local_path = self.target_dir / f"{movie_title}.strm"
                else:
                    # 计算相对于 source_dir 的路径
                    relative_path = bdmv_root.replace(self.source_dir, "", 1)
                    if relative_path.startswith("/"):
                        relative_path = relative_path[1:]
                    
                    # 将 .strm 文件放在电影根目录下，使用电影标题命名
                    local_path = self.target_dir / relative_path / f"{movie_title}.strm"
                
                return local_path

        # 原有逻辑保持不变
        if self.flatten_mode:
            local_path = self.target_dir / path.name
        else:
            relative_path = path.full_path.replace(self.source_dir, "", 1)
            if relative_path.startswith("/"):
                relative_path = relative_path[1:]
            local_path = self.target_dir / relative_path

        if path.suffix.lower() in VIDEO_EXTS:
            local_path = local_path.with_suffix(".strm")

        return local_path

    async def __cleanup_local_files(self) -> None:
        """
        删除服务器中已删除的本地的 .strm 文件及其关联文件
        如果文件后缀在 sync_ignore 中，则不会被删除
        """
        logger.info("开始清理本地文件")

        if self.flatten_mode:
            all_local_files = [f for f in self.target_dir.iterdir() if f.is_file()]
        else:
            all_local_files = [f for f in self.target_dir.rglob("*") if f.is_file()]

        files_to_delete = set(all_local_files) - self.processed_local_paths

        for file_path in files_to_delete:
            # 检查文件是否匹配忽略正则表达式
            if self.sync_ignore_pattern and self.sync_ignore_pattern.search(
                file_path.name
            ):
                logger.debug(f"文件 {file_path.name} 在忽略列表中，跳过删除")
                continue

            try:
                if file_path.exists():
                    await to_thread(file_path.unlink)
                    logger.info(f"删除文件：{file_path}")

                    # 检查并删除空目录
                    parent_dir = file_path.parent
                    while parent_dir != self.target_dir:
                        if any(parent_dir.iterdir()):
                            break  # 目录不为空，跳出循环
                        else:
                            parent_dir.rmdir()
                            logger.info(f"删除空目录：{parent_dir}")
                        parent_dir = parent_dir.parent
            except Exception as e:
                logger.error(f"删除文件 {file_path} 失败：{e}")

    def _is_bdmv_file(self, path: AlistPath) -> bool:
        """
        检查文件是否为 BDMV 结构中的 .m2ts 文件
        
        :param path: AlistPath 对象
        :return: 是否为 BDMV 文件
        """
        return "/BDMV/STREAM/" in path.full_path and path.suffix.lower() == ".m2ts"

    def _get_bdmv_root_dir(self, path: AlistPath) -> str:
        """
        获取 BDMV 文件的根目录路径
        
        :param path: BDMV 中的文件路径
        :return: BDMV 根目录路径
        """
        full_path = path.full_path
        bdmv_index = full_path.find("/BDMV/")
        if bdmv_index != -1:
            return full_path[:bdmv_index]
        return ""

    def _get_movie_title_from_bdmv_path(self, bdmv_root: str) -> str:
        """
        从 BDMV 根目录路径提取电影标题
        
        :param bdmv_root: BDMV 根目录路径
        :return: 电影标题
        """
        # 获取最后一个目录名作为电影标题
        return Path(bdmv_root).name

    def _collect_bdmv_file(self, path: AlistPath) -> None:
        """
        收集 BDMV 文件信息
        
        :param path: BDMV 中的 .m2ts 文件路径
        """
        bdmv_root = self._get_bdmv_root_dir(path)
        if not bdmv_root:
            return

        if bdmv_root not in self.bdmv_collections:
            self.bdmv_collections[bdmv_root] = []

        # 添加文件信息到集合中
        self.bdmv_collections[bdmv_root].append((path, path.size))
        logger.debug(f"收集 BDMV 文件: {path.full_path}, 大小: {path.size}")

    def _finalize_bdmv_collections(self) -> None:
        """
        完成 BDMV 文件收集，确定每个 BDMV 目录中的最大文件
        """
        for bdmv_root, files in self.bdmv_collections.items():
            if not files:
                continue

            movie_title = self._get_movie_title_from_bdmv_path(bdmv_root)
            logger.info(f"BDMV 目录 '{movie_title}' 中发现 {len(files)} 个 .m2ts 文件:")
            
            # 按大小排序并显示所有文件
            sorted_files = sorted(files, key=lambda x: x[1], reverse=True)
            for i, (file_path, file_size) in enumerate(sorted_files):
                size_mb = file_size / (1024 * 1024)
                status = "✓ 选中" if i == 0 else "  跳过"
                logger.info(f"  {status} {file_path.name}: {size_mb:.1f} MB ({file_size} 字节)")

            # 找出最大的文件
            largest_file = max(files, key=lambda x: x[1])
            self.bdmv_largest_files[bdmv_root] = largest_file[0]
            
            largest_size_mb = largest_file[1] / (1024 * 1024)
            logger.info(f"BDMV 目录 '{movie_title}' 最终选择: {largest_file[0].name} ({largest_size_mb:.1f} MB)")

    def _should_process_bdmv_file(self, path: AlistPath) -> bool:
        """
        检查 BDMV 文件是否应该被处理（即是否为最大文件）
        
        :param path: BDMV 中的 .m2ts 文件路径
        :return: 是否应该处理
        """
        bdmv_root = self._get_bdmv_root_dir(path)
        if not bdmv_root:
            return False

        largest_file = self.bdmv_largest_files.get(bdmv_root)
        return largest_file is not None and largest_file.full_path == path.full_path


```

## `/Users/yuanjing/Code/AutoFilm/app/modules/alist2strm/__init__.py`

```
from app.modules.alist2strm.alist2strm import Alist2Strm
```

## `/Users/yuanjing/Code/AutoFilm/app/modules/alist/v3/storage.py`

```
from json import loads, dumps
from typing import Literal
from types import FunctionType

from pydantic import BaseModel, ConfigDict, model_validator


class AlistStorage(BaseModel):
    """
    Alist 存储器模型
    """

    model_config = ConfigDict(
        ignored_types=(FunctionType, type(lambda: None))  # 覆盖 Cython 类型
    )

    id: int = 0  # 存储器 ID
    status: Literal["work", "disabled"] = "work"  # 存储器状态
    remark: str = ""  # 备注
    modified: str = ""  # 修改时间
    disabled: bool = False  # 是否禁用
    mount_path: str = ""  # 挂载路径
    order: int = 0  # 排序
    driver: str = "Local"  # 驱动器
    cache_expiration: int = 30  # 缓存过期时间
    addition: str = "{}"  # 附加信息
    enable_sign: bool = False  # 是否启用签名
    order_by: str = "name"  # 排序字段
    order_direction: str = "asc"  # 排序方向
    extract_folder: str = "front"  # 提取文件夹
    web_proxy: bool = False  # 是否启用 Web 代理
    webdav_policy: str = "native_proxy"  # WebDAV 策略
    down_proxy_url: str = ""  # 下载代理 URL

    def set_addition_by_dict(self, additon: dict) -> None:
        """
        使用 Python 字典设置 Storage 附加信息
        """
        self.addition = dumps(additon)

    @property
    def addition2dict(self) -> dict:
        """
        获取 Storage 附加信息，返回Python 字典
        """
        return loads(self.addition)

    @model_validator(mode="before")
    def check_status(cls, values: dict) -> dict:
        status = values.get("status")
        disabled = values.get("disabled")
        if (disabled and status == "work") or (not disabled and status == "disabled"):
            raise ValueError(f"存储器状态错误，{status=}, {disabled=}")
        return values


if __name__ == "__main__":
    info = {
        "id": 1,
        "mount_path": "/lll",
        "order": 0,
        "driver": "Local",
        "cache_expiration": 0,
        "status": "work",
        "addition": '{"root_folder_path":"/root/www","thumbnail":false,"thumb_cache_folder":"","show_hidden":true,"mkdir_perm":"777"}',
        "remark": "",
        "modified": "2023-07-19T09:46:38.868739912+08:00",
        "disabled": False,
        "enable_sign": False,
        "order_by": "name",
        "order_direction": "asc",
        "extract_folder": "front",
        "web_proxy": False,
        "webdav_policy": "native_proxy",
        "down_proxy_url": "",
    }
    storage = AlistStorage(**info)
    print(storage)
    print(storage.addition2dict)
    storage.set_addition_by_dict({"test": 1})
    print(storage.addition)
    print(storage.addition2dict)
    storage.addition = '{"test": 2}'
    print(storage.addition)
    print(storage.addition2dict)

```

## `/Users/yuanjing/Code/AutoFilm/app/modules/alist/v3/path.py`

```
from re import sub
from typing import Any
from datetime import datetime

from pydantic import BaseModel

from app.utils import URLUtils


class AlistPath(BaseModel):
    """
    Alist 文件/目录对象
    """

    server_url: str  # 服务器地址
    base_path: str  # 用户基础路径（用于计算文件/目录在 Alist 服务器上的绝对地址）
    full_path: str  # 相对用户根文件/目录路径

    id: str | None = None  # 文件/目录 ID（Alist V3.45）
    path: str | None = None  # 相对存储器根目录的文件/目录路径（Alist V3.45）
    name: str  # 文件/目录名称
    size: int  # 文件大小
    is_dir: bool  # 是否为目录
    modified: str  # 修改时间
    created: str  # 创建时间
    sign: str  # 签名
    thumb: str  # 缩略图
    type: int  # 类型
    hashinfo: str  # 哈希信息（字符串）
    hash_info: dict | None = None  # 哈希信息（键值对）

    # g/api/fs/get 返回新增的字段（详细信息）
    raw_url: str | None = None  # 原始地址
    readme: str | None = None  # Readme 地址
    header: str | None = None  # 头部信息
    provider: str | None = None  # 提供者
    related: Any = None  # 相关信息

    @property
    def abs_path(self) -> str:
        """
        文件/目录在 Alist 服务器上的绝对路径
        """
        return self.base_path.rstrip("/") + self.full_path

    @property
    def download_url(self) -> str:
        """
        文件下载地址
        """
        if self.sign:
            url = self.server_url + "/d" + self.abs_path + "?sign=" + self.sign
        else:
            url = self.server_url + "/d" + self.abs_path

        return URLUtils.encode(url)

    @property
    def proxy_download_url(self) -> str:
        """
        Alist代理下载地址
        """
        return sub("/d/", "/p/", self.download_url, 1)

    @property
    def suffix(self) -> str:
        """
        文件后缀
        """
        if self.is_dir:
            return ""
        else:
            return "." + self.name.split(".")[-1]

    def __parse_timestamp(self, time_str: str) -> float:
        """
        解析时间字符串得到时间的时间戳
        """
        dt = datetime.fromisoformat(time_str)
        return dt.timestamp()

    @property
    def modified_timestamp(self) -> float:
        """
        获得修改时间的时间戳
        """
        return self.__parse_timestamp(self.modified)

    @property
    def created_timestamp(self) -> float:
        """
        获得创建时间的时间戳
        """
        return self.__parse_timestamp(self.created)

```

## `/Users/yuanjing/Code/AutoFilm/app/modules/alist/v3/client.py`

```
from asyncio import sleep
from typing import Callable, AsyncGenerator
from time import time

from httpx import Response

from app.core import logger
from app.utils import RequestUtils, Multiton
from app.modules.alist.v3.path import AlistPath
from app.modules.alist.v3.storage import AlistStorage


class AlistClient(metaclass=Multiton):
    """
    Alist 客户端 API
    """

    def __init__(
        self,
        url: str,
        username: str = "",
        password: str = "",
        token: str = "",
    ) -> None:
        """
        AlistClient 类初始化

        :param url: Alist 服务器地址
        :param username: Alist 用户名
        :param password: Alist 密码
        :param token: Alist 永久令牌
        """

        if (username == "" or password == "") and token == "":
            raise ValueError("用户名及密码为空或令牌 Token 为空")

        self.__client = RequestUtils.get_client()
        self.__token = {
            "token": "",  # 令牌 token str
            "expires": 0,  # 令牌过期时间（时间戳，-1为永不过期） int
        }
        self.base_path = ""
        self.id = 0

        if not url.startswith("http"):
            url = "https://" + url
        self.url = url.rstrip("/")

        if token != "":
            self.__token["token"] = token
            self.__token["expires"] = -1
        elif username != "" and password != "":
            self.__username = str(username)
            self.___password = str(password)
        else:
            raise ValueError("用户名及密码为空或令牌 Token 为空")

        self.sync_api_me()

    async def __request(
        self,
        method: str,
        url: str,
        auth: bool = True,
        **kwargs,
    ) -> Response:
        """
        发送 HTTP 请求

        :param method 请求方法
        :param url 请求 url
        :param auth header 中是否带有 alist 认证令牌
        """

        if auth:
            headers = kwargs.get("headers", {})
            headers["Authorization"] = self.__get_token
            kwargs["headers"] = headers
        return await self.__client.request(method, url, **kwargs, sync=False)

    async def __get(self, url: str, auth: bool = True, **kwargs) -> Response:
        """
        发送 GET 请求

        :param url 请求 url
        :param auth header 中是否带有 alist 认证令牌
        """
        return await self.__request("get", url, auth, **kwargs)

    async def __post(self, url: str, auth: bool = True, **kwargs) -> Response:
        """
        发送 POST 请求

        :param url 请求 url
        :param auth header 中是否带有 alist 认证令牌
        """
        return await self.__request("post", url, auth, **kwargs)

    @property
    def username(self) -> str:
        """
        获取用户名
        """

        return self.__username

    @property
    def __password(self) -> str:
        """
        获取密码
        """

        return self.___password

    @property
    def __get_token(self) -> str:
        """
        返回可用登录令牌

        :return: 登录令牌 token
        """

        if self.__token["expires"] == -1:
            logger.debug("使用永久令牌")
            return self.__token["token"]
        else:
            logger.debug("使用临时令牌")
            now_stamp = int(time())

            if self.__token["expires"] < now_stamp:  # 令牌过期需要重新更新
                self.__token["token"] = self.api_auth_login()
                self.__token["expires"] = (
                    now_stamp + 2 * 24 * 60 * 60 - 5 * 60
                )  # 2天 - 5分钟（alist 令牌有效期为 2 天，提前 5 分钟刷新）

            return self.__token["token"]

    def api_auth_login(self) -> str:
        """
        登录 Alist 服务器认证账户信息

        :return: 重新申请的登录令牌 token
        """

        json = {"username": self.username, "password": self.__password}
        resp = self.__client.post(self.url + "/api/auth/login", json=json, sync=True)
        if resp.status_code != 200:
            raise RuntimeError(f"更新令牌请求发送失败，状态码：{resp.status_code}")

        result = resp.json()

        if result["code"] != 200:
            raise RuntimeError(f"更新令牌，错误信息：{result['message']}")

        logger.debug(f"{self.username} 更新令牌成功")
        return result["data"]["token"]

    def sync_api_me(self) -> None:
        """
        获取用户信息
        获取当前用户 base_path 和 id 并分别保存在 self.base_path 和 self.id 中
        """

        headers = {"Authorization": self.__get_token}
        resp = self.__client.get(self.url + "/api/me", headers=headers, sync=True)

        if resp.status_code != 200:
            raise RuntimeError(f"获取用户信息请求发送失败，状态码：{resp.status_code}")

        result = resp.json()

        if result["code"] != 200:
            raise RuntimeError(f"获取用户信息失败，错误信息：{result['message']}")

        try:
            self.base_path: str = result["data"]["base_path"]
            self.id: int = result["data"]["id"]
        except Exception:
            raise RuntimeError("获取用户信息失败")

    async def async_api_fs_list(self, dir_path: str) -> list[AlistPath]:
        """
        获取文件列表

        :param dir_path: 目录路径
        :return: AlistPath 对象列表
        """

        logger.debug(f"获取目录 {dir_path} 下的文件列表")

        json = {
            "path": dir_path,
            "password": "",
            "page": 1,
            "per_page": 0,
            "refresh": False,
        }

        resp = await self.__post(self.url + "/api/fs/list", json=json)
        if resp.status_code != 200:
            raise RuntimeError(
                f"获取目录 {dir_path} 的文件列表请求发送失败，状态码：{resp.status_code}"
            )

        result = resp.json()

        if result["code"] != 200:
            raise RuntimeError(
                f"获取目录 {dir_path} 的文件列表失败，错误信息：{result['message']}"
            )

        logger.debug(f"获取目录 {dir_path} 的文件列表成功")

        if result["data"]["total"] == 0:
            return []

        return [
            AlistPath(
                server_url=self.url,
                base_path=self.base_path,
                full_path=dir_path + "/" + alist_path["name"],
                **alist_path,
            )
            for alist_path in result["data"]["content"]
        ]

    async def async_api_fs_get(self, path: str) -> AlistPath:
        """
        获取文件/目录详细信息

        :param path: 文件/目录路径
        :return: AlistPath 对象
        """

        json = {
            "path": path,
            "password": "",
            "page": 1,
            "per_page": 0,
            "refresh": False,
        }

        resp = await self.__post(self.url + "/api/fs/get", json=json)
        if resp.status_code != 200:
            raise RuntimeError(
                f"获取路径 {path} 详细信息请求发送失败，状态码：{resp.status_code}"
            )
        result = resp.json()

        if result["code"] != 200:
            raise RuntimeError(
                f"获取路径 {path} 详细信息失败，详细信息：{result['message']}"
            )

        logger.debug(f"获取路径 {path} 详细信息成功")
        return AlistPath(
            server_url=self.url,
            base_path=self.base_path,
            full_path=path,
            **result["data"],
        )

    async def async_api_admin_storage_list(self) -> list[AlistStorage]:
        """
        列出存储列表 需要管理员用户权限

        :return: AlistStorage 对象列表
        """

        resp = await self.__get(self.url + "/api/admin/storage/list")
        if resp.status_code != 200:
            raise RuntimeError(
                f"获取存储器列表请求发送失败，状态码：{resp.status_code}"
            )

        result = resp.json()

        if result["code"] != 200:
            raise RuntimeError(f"获取存储器列表失败，详细信息：{result['message']}")

        logger.debug("获取存储器列表成功")
        return [AlistStorage(**storage) for storage in result["data"]["content"]]

    async def async_api_admin_storage_create(self, storage: AlistStorage) -> None:
        """
        创建存储 需要管理员用户权限

        :param storage: AlistStorage 对象
        """

        json = {
            "mount_path": storage.mount_path,
            "order": storage.order,
            "remark": storage.remark,
            "cache_expiration": storage.cache_expiration,
            "web_proxy": storage.web_proxy,
            "webdav_policy": storage.webdav_policy,
            "down_proxy_url": storage.down_proxy_url,
            "enable_sign": storage.enable_sign,
            "driver": storage.driver,
            "order_by": storage.order_by,
            "order_direction": storage.order_direction,
            "addition": storage.addition,
        }

        resp = await self.__post(self.url + "/api/admin/storage/create", json=json)
        if resp.status_code != 200:
            raise RuntimeError(f"创建存储请求发送失败，状态码：{resp.status_code}")
        result = resp.json()

        if result["code"] != 200:
            raise RuntimeError(f"创建存储失败，详细信息：{result['message']}")

        logger.debug("创建存储成功")
        return

    async def async_api_admin_storage_update(self, storage: AlistStorage) -> None:
        """
        更新存储，需要管理员用户权限

        :param storage: AlistStorage 对象
        """

        json = {
            "id": storage.id,
            "mount_path": storage.mount_path,
            "order": storage.order,
            "driver": storage.driver,
            "cache_expiration": storage.cache_expiration,
            "status": storage.status,
            "addition": storage.addition,
            "remark": storage.remark,
            "modified": storage.modified,
            "disabled": storage.disabled,
            "enable_sign": storage.enable_sign,
            "order_by": storage.order_by,
            "order_direction": storage.order_direction,
            "extract_folder": storage.extract_folder,
            "web_proxy": storage.web_proxy,
            "webdav_policy": storage.webdav_policy,
            "down_proxy_url": storage.down_proxy_url,
        }

        resp = await self.__post(self.url + "/api/admin/storage/update", json=json)
        if resp.status_code != 200:
            raise RuntimeError(f"更新存储请求发送失败，状态码：{resp.status_code}")

        result = resp.json()

        if result["code"] != 200:
            raise RuntimeError(f"更新存储器失败，详细信息：{result['message']}")

        logger.debug(
            f"更新存储器成功，存储器ID：{storage.id}，挂载路径：{storage.mount_path}"
        )
        return

    async def iter_path(
        self,
        dir_path: str,
        wait_time: float | int,
        is_detail: bool = True,
        filter: Callable[[AlistPath], bool] = lambda x: True,
    ) -> AsyncGenerator[AlistPath, None]:
        """
        异步路径列表生成器
        返回目录及其子目录的所有文件和目录的 AlistPath 对象

        :param dir_path: 目录路径
        :param wait_time: 每轮遍历等待时间（单位秒）,
        :param is_detail：是否获取详细信息（raw_url）
        :param filter: 匿名函数过滤器（默认不启用）
        :return: AlistPath 对象生成器
        """

        for path in await self.async_api_fs_list(dir_path):
            await sleep(wait_time)
            if path.is_dir:
                async for child_path in self.iter_path(
                    dir_path=path.full_path,
                    wait_time=wait_time,
                    is_detail=is_detail,
                    filter=filter,
                ):
                    yield child_path

            if filter(path):
                if is_detail:
                    yield await self.async_api_fs_get(path.full_path)
                else:
                    yield path

    async def get_storage_by_mount_path(
        self, mount_path: str, create: bool = False, **kwargs
    ) -> AlistStorage | None:
        """
        通过挂载路径获取存储器信息

        :param mount_path: 挂载路径
        :param create: 未找到存储器时是否创建
        :param kwargs: 创建存储器 AlistStorge 时的参数
        :return: AlistStorage 对象
        """

        for storage in await self.async_api_admin_storage_list():
            if storage.mount_path == mount_path:
                return storage
        logger.debug(f"在 Alist 服务器上未找到存储器 {mount_path}")

        if create:
            kwargs["mount_path"] = mount_path
            storage = AlistStorage(**kwargs)
            await self.async_api_admin_storage_create(storage)
            return storage

        return None

```

## `/Users/yuanjing/Code/AutoFilm/app/modules/alist/v3/__init__.py`

```
from app.modules.alist.v3.client import AlistClient
from app.modules.alist.v3.path import AlistPath
from app.modules.alist.v3.storage import AlistStorage

```

## `/Users/yuanjing/Code/AutoFilm/app/modules/alist/__init__.py`

```
"""
Alist API V3
https://alist.nn.ci/zh/guide/api/
"""

from app.modules.alist.v3 import AlistClient, AlistPath, AlistStorage

```

## `/Users/yuanjing/Code/AutoFilm/app/modules/__init__.py`

```
from app.modules.alist2strm import Alist2Strm
from app.modules.ani2alist import Ani2Alist
from app.modules.libraryposter import LibraryPoster

__all__ = [
    Alist2Strm,
    Ani2Alist,
    LibraryPoster,
]

```

## `/Users/yuanjing/Code/AutoFilm/app/main.py`

```
from asyncio import get_event_loop
from sys import path
from os.path import dirname

path.append(dirname(dirname(__file__)))

from apscheduler.schedulers.asyncio import AsyncIOScheduler  # type:ignore
from apscheduler.triggers.cron import CronTrigger  # type:ignore

from app.core import settings, logger
from app.extensions import LOGO
from app.modules import Alist2Strm, Ani2Alist, LibraryPoster


def print_logo() -> None:
    """
    打印 Logo
    """

    print(LOGO)
    print(f" {settings.APP_NAME} {settings.APP_VERSION} ".center(65, "="))
    print("")


if __name__ == "__main__":
    print_logo()

    logger.info(f"AutoFilm {settings.APP_VERSION} 启动中...")
    logger.debug(f"是否开启 DEBUG 模式: {settings.DEBUG}")

    scheduler = AsyncIOScheduler()

    if settings.AlistServerList:
        logger.info("检测到 Alist2Strm 模块配置，正在添加至后台任务")
        for server in settings.AlistServerList:
            cron = server.get("cron")
            if cron:
                scheduler.add_job(
                    Alist2Strm(**server).run, trigger=CronTrigger.from_crontab(cron)
                )
                logger.info(f"{server['id']} 已被添加至后台任务")
            else:
                logger.warning(f"{server['id']} 未设置 cron")
    else:
        logger.warning("未检测到 Alist2Strm 模块配置")

    if settings.Ani2AlistList:
        logger.info("检测到 Ani2Alist 模块配置，正在添加至后台任务")
        for server in settings.Ani2AlistList:
            cron = server.get("cron")
            if cron:
                scheduler.add_job(
                    Ani2Alist(**server).run, trigger=CronTrigger.from_crontab(cron)
                )
                logger.info(f"{server['id']} 已被添加至后台任务")
            else:
                logger.warning(f"{server['id']} 未设置 cron")
    else:
        logger.warning("未检测到 Ani2Alist 模块配置")

    if settings.LibraryPosterList:
        logger.info("检测到 LibraryPoster 模块配置，正在添加至后台任务")
        for poster in settings.LibraryPosterList:
            cron = poster.get("cron")
            if cron:
                scheduler.add_job(
                    LibraryPoster(**poster).run, trigger=CronTrigger.from_crontab(cron)
                )
                logger.info(f"{poster['id']} 已被添加至后台任务")
            else:
                logger.warning(f"{poster['id']} 未设置 cron")
    else:
        logger.warning("未检测到 LibraryPoster 模块配置")

    scheduler.start()
    logger.info("AutoFilm 启动完成")

    try:
        get_event_loop().run_forever()
    except (KeyboardInterrupt, SystemExit):
        logger.info("AutoFilm 程序退出！")

```

## `/Users/yuanjing/Code/AutoFilm/app/extensions/media/releasegroup.py`

```
from typing import Final

# 电影字幕组
MOVIE_RELEASEGROUP: Final = frozenset()

# 电视剧字幕组
TV_RELEASEGROUP: Final = frozenset()

# 动漫字幕组
ANIEME_RELEASEGROUP: Final = frozenset(
    (
        "ANi",
        "HYSUB",
        "KTXP",
        "LoliHouse",
        "MCE",
        "Nekomoe kissaten",
        "SweetSub",
        "MingY",
        "(?:Lilith|NC)-Raws",
        "织梦字幕组",
        "枫叶字幕组",
        "猎户手抄部",
        "喵萌奶茶屋",
        "漫猫字幕社",
        "霜庭云花Sub",
        "北宇治字幕组",
        "氢气烤肉架",
        "云歌字幕组",
        "萌樱字幕组",
        "极影字幕社",
        "悠哈璃羽字幕社",
        "❀拨雪寻春❀",
        "沸羊羊(?:制作|字幕组)",
        "(?:桜|樱)都字幕组",
    )
)

# 未分类字幕组
OTHER_RELEASEGROUP: Final = frozenset(())

RELEASEGROUP = MOVIE_RELEASEGROUP | TV_RELEASEGROUP | ANIEME_RELEASEGROUP | OTHER_RELEASEGROUP
```

## `/Users/yuanjing/Code/AutoFilm/app/extensions/media/__init__.py`

```
from app.extensions.media.releasegroup import RELEASEGROUP
```

## `/Users/yuanjing/Code/AutoFilm/app/extensions/logo.py`

```
LOGO = r"""
 █████╗ ██╗   ██╗████████╗ ██████╗ ███████╗██╗██╗     ███╗   ███╗    
██╔══██╗██║   ██║╚══██╔══╝██╔═══██╗██╔════╝██║██║     ████╗ ████║    
███████║██║   ██║   ██║   ██║   ██║█████╗  ██║██║     ██╔████╔██║    
██╔══██║██║   ██║   ██║   ██║   ██║██╔══╝  ██║██║     ██║╚██╔╝██║    
██║  ██║╚██████╔╝   ██║   ╚██████╔╝██║     ██║███████╗██║ ╚═╝ ██║    
╚═╝  ╚═╝ ╚═════╝    ╚═╝    ╚═════╝ ╚═╝     ╚═╝╚══════╝╚═╝     ╚═╝
"""
```

## `/Users/yuanjing/Code/AutoFilm/app/extensions/exts.py`

```
from typing import Final

VIDEO_EXTS: Final = frozenset(
    (
        ".mp4",
        ".mkv",
        ".flv",
        ".avi",
        ".wmv",
        ".ts",
        ".rmvb",
        ".webm",
        ".wmv",
        ".mpg",
        ".m2ts",
    )
)  # 视频文件后缀
EXTENDED_VIDEO_EXTS: Final = VIDEO_EXTS.union((".strm",))  # 扩展视频文件后缀

SUBTITLE_EXTS: Final = frozenset((".ass", ".srt", ".ssa", ".sub"))  # 字幕文件后缀

IMAGE_EXTS: Final = frozenset((".png", ".jpg"))

NFO_EXTS: Final = frozenset((".nfo",))

```

## `/Users/yuanjing/Code/AutoFilm/app/extensions/__init__.py`

```
from app.extensions.exts import (
    VIDEO_EXTS,
    EXTENDED_VIDEO_EXTS,
    SUBTITLE_EXTS,
    IMAGE_EXTS,
    NFO_EXTS,
)
from app.extensions.logo import LOGO

```

## `/Users/yuanjing/Code/AutoFilm/app/core/log.py`

```
import logging
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime
from pathlib import Path

import click

from app.core.config import settings

FMT = "%(prefix)s %(message)s"

# 日志级别颜色映射
LEVEL_WITH_COLOR = {
    logging.DEBUG: lambda level_name: click.style(str(level_name), fg="blue"),
    logging.INFO: lambda level_name: click.style(str(level_name), fg="green"),
    logging.WARNING: lambda level_name: click.style(str(level_name), fg="yellow"),
    logging.ERROR: lambda level_name: click.style(str(level_name), fg="red"),
    logging.CRITICAL: lambda level_name: click.style(
        str(level_name), fg="red", bold=True
    ),
}


class CustomFormatter(logging.Formatter):
    """
    自定义日志输出格式

    对 logging.LogRecord 增加一个属性 prefix，level + time
    """

    def __init__(self, file_formatter: bool = False, fmt: str = None) -> None:
        """
        :param file_formatter: 是否为文件格式化器
        """

        self.__file_formatter = file_formatter
        super().__init__(fmt=fmt)

    def format(self, record: logging.LogRecord) -> str:

        if self.__file_formatter:  # 文件中不需要控制字
            record.prefix = f"【{record.levelname}】"
        else:  # 控制台需要控制字
            record.prefix = LEVEL_WITH_COLOR[record.levelno](f"【{record.levelname}】")

        # 最长的 CRITICAL 为 8 个字符，保留 1 个空格作为分隔符
        separator = " " * (9 - len(record.levelname))
        dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        record.prefix += f"{separator}{dt} |"
        return super().format(record)


class TRFileHandler(TimedRotatingFileHandler):
    """
    日期轮换文件处理器
    """

    def __init__(self, log_dir: Path, encoding: str = "utf-8") -> None:
        self.log_dir = log_dir
        super().__init__(
            self.__get_log_filname(),
            when="midnight",
            interval=1,
            backupCount=0,
            encoding=encoding,
        )

    def doRollover(self) -> None:
        """
        在轮换日志文件时，更新日志文件路径
        """

        self.baseFilename = self.__get_log_filname()
        super().doRollover()

    def __get_log_filname(self) -> str:
        """
        根据当前日期生成日志文件路径
        """

        current_date = datetime.now().strftime("%Y-%m-%d")
        return (self.log_dir / f"{current_date}.log").as_posix()


class LoggerManager:
    """
    日志管理器
    """

    def __init__(self) -> None:
        """
        初始化 LoggerManager 对象
        """

        self.__logger = logging.getLogger(settings.APP_NAME)
        self.__logger.setLevel(logging.DEBUG)

        console_formatter = CustomFormatter(
            file_formatter=False,
            fmt=FMT,
        )
        file_formatter = CustomFormatter(
            file_formatter=True,
            fmt=FMT,
        )

        level = logging.DEBUG if settings.DEBUG else logging.INFO

        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(console_formatter)
        self.__logger.addHandler(console_handler)

        file_handler = TRFileHandler(log_dir=settings.LOG_DIR, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(file_formatter)
        self.__logger.addHandler(file_handler)

    def __log(self, method: str, msg: str, *args, **kwargs) -> None:
        """
        获取模块的logger
        :param method: 日志方法
        :param msg: 日志信息
        """
        if hasattr(self.__logger, method):
            getattr(self.__logger, method)(msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs) -> None:
        """
        重载info方法
        """
        self.__log("info", msg, *args, **kwargs)

    def debug(self, msg: str, *args, **kwargs) -> None:
        """
        重载debug方法
        """
        self.__log("debug", msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs) -> None:
        """
        重载warning方法
        """
        self.__log("warning", msg, *args, **kwargs)

    def warn(self, msg: str, *args, **kwargs) -> None:
        """
        重载warn方法
        """
        self.__log("warning", msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs) -> None:
        """
        重载error方法
        """
        self.__log("error", msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs) -> None:
        """
        重载critical方法
        """
        self.__log("critical", msg, *args, **kwargs)


# 初始化公共日志
logger = LoggerManager()

```

## `/Users/yuanjing/Code/AutoFilm/app/core/config.py`

```
from pathlib import Path
from yaml import safe_load
from typing import Any

from app.version import APP_VERSION


class SettingManager:
    """
    系统配置
    """

    # APP 名称
    APP_NAME: str = "Autofilm"
    # APP 版本
    APP_VERSION: str = APP_VERSION
    # 时区
    TZ: str = "Asia/Shanghai"
    # 开发者模式
    DEBUG: bool = False

    def __init__(self) -> None:
        """
        初始化 SettingManager 对象
        """
        self.__mkdir()
        self.__load_mode()

    def __mkdir(self) -> None:
        """
        创建目录
        """
        with self.CONFIG_DIR as dir_path:
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)

        with self.LOG_DIR as dir_path:
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)

    def __load_mode(self) -> None:
        """
        加载模式
        """
        with self.CONFIG.open(mode="r", encoding="utf-8") as file:
            is_dev = safe_load(file).get("Settings", {}).get("DEV", False)

        self.DEBUG = is_dev

    @property
    def BASE_DIR(self) -> Path:
        """
        后端程序基础路径 AutoFilm/app
        """
        return Path(__file__).parents[2]

    @property
    def CONFIG_DIR(self) -> Path:
        """
        配置文件路径
        """
        return self.BASE_DIR / "config"

    @property
    def LOG_DIR(self) -> Path:
        """
        日志文件路径
        """
        return self.BASE_DIR / "logs"

    @property
    def CONFIG(self) -> Path:
        """
        配置文件
        """
        return self.CONFIG_DIR / "config.yaml"

    @property
    def LOG(self) -> Path:
        """
        日志文件
        """
        if self.DEBUG:
            return self.LOG_DIR / "dev.log"
        else:
            return self.LOG_DIR / "AutoFilm.log"

    @property
    def AlistServerList(self) -> list[dict[str, Any]]:
        with self.CONFIG.open(mode="r", encoding="utf-8") as file:
            alist_server_list = safe_load(file).get("Alist2StrmList", [])
        return alist_server_list

    @property
    def Ani2AlistList(self) -> list[dict[str, Any]]:
        with self.CONFIG.open(mode="r", encoding="utf-8") as file:
            ani2alist_list = safe_load(file).get("Ani2AlistList", [])
        return ani2alist_list

    @property
    def LibraryPosterList(self) -> list[dict[str, Any]]:
        with self.CONFIG.open(mode="r", encoding="utf-8") as file:
            library_poster_list = safe_load(file).get("LibraryPosterList", [])
        return library_poster_list


settings = SettingManager()

```

## `/Users/yuanjing/Code/AutoFilm/app/core/__init__.py`

```
from app.core.config import settings
from app.core.log import logger
```

## `/Users/yuanjing/Code/AutoFilm/README.md`

```
[license]: /LICENSE
[license-badge]: https://img.shields.io/github/license/Akimio521/AutoFilm?style=flat-square&a=1
[prs]: https://github.com/Akimio521/AutoFilm
[prs-badge]: https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square
[issues]: https://github.com/Akimio521/AutoFilm/issues/new
[issues-badge]: https://img.shields.io/badge/Issues-welcome-brightgreen.svg?style=flat-square
[release]: https://github.com/Akimio521/AutoFilm/releases/latest
[release-badge]: https://img.shields.io/github/v/release/Akimio521/AutoFilm?style=flat-square
[docker]: https://hub.docker.com/r/akimio/autofilm
[docker-badge]: https://img.shields.io/docker/pulls/akimio/autofilm?color=%2348BB78&logo=docker&label=pulls

<div align="center">

# AutoFilm

**一个为 Emby、Jellyfin 服务器提供直链播放的小项目** 

[![license][license-badge]][license]
[![prs][prs-badge]][prs]
[![issues][issues-badge]][issues]
[![release][release-badge]][release]
[![docker][docker-badge]][docker]


[说明文档](#说明文档) •
[部署方式](#部署方式) •
[Strm文件优点](#Strm文件优点) •
[TODO LIST](#todo-list) •
[更新日志](#更新日志) •
[贡献者](#贡献者) •
[Star History](#star-history)

</div>

# 说明文档
详情见 [AutoFilm 说明文档](https://blog.akimio.top/posts/1031/)

# 部署方式
1. Docker 运行
    ```bash
    docker run -d --name autofilm  -v ./config:/config -v ./media:/media -v ./logs:/logs akimio/autofilm
    ```
2. Python 环境运行（Python3.12）
    ```bash
    python app/main.py
    ```

# Strm文件优点
- [x] 轻量化 Emby 服务器，降低 Emby 服务器的性能需求以及硬盘需求
- [x] 运行稳定
- [x] 相比直接访问 Webdav，Emby、Jellyfin 服务器可以提供更好的视频搜索功能以及自带刮削器，以及多设备同步播放进度
- [x] 提高访问速度，播放速度不受 Emby / Jellyfin 服务器带宽限制（需要使用 [MediaWarp](https://github.com/Akimio521/MediaWarp)）

# TODO LIST
- [x] 从 config 文件中读取配置
- [x] 优化程序运行效率（异步处理）
- [x] 增加 Docker 镜像
- [x] 本地同步网盘
- [x] Alist 永久令牌
- [x] LibraryPoster（媒体库海报，感谢[HappyQuQu/jellyfin-library-poster](https://github.com/HappyQuQu/jellyfin-library-poster)）
- [ ] 使用 API 触发任务
- [ ] 通知功能
- [ ] ~~对接 TMDB 实现分类、重命名、刮削等功能~~
    > 已经向 [MoviePilot](https://github.com/jxxghp/MoviePilot) 提交支持对 Alist 服务器文件的操作功能的 PR，目前已经合并进入主线分支，可以直接使用 MoviePilot 直接刮削

# 功能演示
## LibraryPoster
美化媒体库海报封面图

![LibraryPoster](./img/LibraryPoster.png)

# 更新日志
- 2025.7.14：v1.4.0，修复 Ani2Alist 模块时间解析问题，新增 LibraryPoster 美化媒体库封面模块
- 2025.5.29：v1.3.3，Alist2Strm 模块支持添加删除空目录的功能；提高 Alist V3.45 兼容性；添加 m2ts 视频文件后缀到视频扩展集合；修复视频扩展集合中".wmv"缺失前缀错误
- 2025.4.4：v1.3.2，添加 .mpg 视频文件后缀；优化重试装饰器；优化重试装饰器；新增遍历文件间隔时间，防止被风控；修正部分方法名、返回变量类型、文档表述错误
- 2025.3.15：v1.3.1，修复重试装饰器参数类型错误；在 AlistStorage 中添加 model_config 以忽略特定类型避免 Cython 编译后无法使用；修改 AlistClient 中的异常捕获以避免捕获其他异常；使用 Cython 对 Docker 容器内的 py 文件编译，提高性能
- 2025.3.12：v1.3.0，增加汉字转拼音相关工具；修复 AlistStorage 属性调用错误问题；修复 RSS 订阅更新对 storage.addition2dict 结构中 url_structure 的处理；修复无法仅 token 实例化 AlistClient 对象问题；优化 Ani2Alist 运行逻辑；优化 Ani2Alist 性能，减少 URL 解码次数；优化 Alist2Strm 支持判断本地文件是否过期或损坏而进行重新处理
- 2025.1.10：v1.2.6 使用 RequestUtils 作为全局统一的 HTTP 请求出口、更新 Docker 镜像底包、Alist2Strm 新增同步忽略功能
- 2024.11.8：v1.2.5，Alist2Strm 模块新增同步功能；优化 AlistClient，减少 token 申请；支持使用永久令牌；优化日志功能
- 2024.8.26：v1.2.4，完善 URL 中文字符编码问题；提高 Python3.11 兼容性；Alist2Strm 的 mode 选项
- 2024.7.17：v1.2.2，增加 Ani2Strm 模块
- 2024.7.8：v1.2.0，修改程序运行逻辑，使用 AsyncIOScheduler 实现后台定时任务
- 2024.6.3：v1.1.0，使用 alist 官方 api 替代 webdav 实现“扫库”；采用异步并发提高运行效率；配置文件有改动；支持非基础路径 Alist 用户以及无 Webdav 权限用户
- 2024.5.29：v1.0.2，优化运行逻辑；Docker 部署，自动打包 Docker 镜像
- 2024.2.1：v1.0.0，完全重构 AutoFilm ，不再兼容 v0.1 ；实现多线程，大幅度提升任务处理速度
- 2024.1.28：v0.1.1，初始版本持续迭代

# 贡献者
<a href="https://github.com/Akimio521/AutoFilm/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Akimio521/AutoFilm" />
</a>

# Star History
<a href="https://github.com/Akimio521/AutoFilm/stargazers">
    <img width="500" alt="Star History Chart" src="https://api.star-history.com/svg?repos=Akimio521/AutoFilm&type=Date">
</a> 
```

## `/Users/yuanjing/Code/AutoFilm/LICENSE`

```
                    GNU AFFERO GENERAL PUBLIC LICENSE
                       Version 3, 19 November 2007

 Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.

                            Preamble

  The GNU Affero General Public License is a free, copyleft license for
software and other kinds of works, specifically designed to ensure
cooperation with the community in the case of network server software.

  The licenses for most software and other practical works are designed
to take away your freedom to share and change the works.  By contrast,
our General Public Licenses are intended to guarantee your freedom to
share and change all versions of a program--to make sure it remains free
software for all its users.

  When we speak of free software, we are referring to freedom, not
price.  Our General Public Licenses are designed to make sure that you
have the freedom to distribute copies of free software (and charge for
them if you wish), that you receive source code or can get it if you
want it, that you can change the software or use pieces of it in new
free programs, and that you know you can do these things.

  Developers that use our General Public Licenses protect your rights
with two steps: (1) assert copyright on the software, and (2) offer
you this License which gives you legal permission to copy, distribute
and/or modify the software.

  A secondary benefit of defending all users' freedom is that
improvements made in alternate versions of the program, if they
receive widespread use, become available for other developers to
incorporate.  Many developers of free software are heartened and
encouraged by the resulting cooperation.  However, in the case of
software used on network servers, this result may fail to come about.
The GNU General Public License permits making a modified version and
letting the public access it on a server without ever releasing its
source code to the public.

  The GNU Affero General Public License is designed specifically to
ensure that, in such cases, the modified source code becomes available
to the community.  It requires the operator of a network server to
provide the source code of the modified version running there to the
users of that server.  Therefore, public use of a modified version, on
a publicly accessible server, gives the public access to the source
code of the modified version.

  An older license, called the Affero General Public License and
published by Affero, was designed to accomplish similar goals.  This is
a different license, not a version of the Affero GPL, but Affero has
released a new version of the Affero GPL which permits relicensing under
this license.

  The precise terms and conditions for copying, distribution and
modification follow.

                       TERMS AND CONDITIONS

  0. Definitions.

  "This License" refers to version 3 of the GNU Affero General Public License.

  "Copyright" also means copyright-like laws that apply to other kinds of
works, such as semiconductor masks.

  "The Program" refers to any copyrightable work licensed under this
License.  Each licensee is addressed as "you".  "Licensees" and
"recipients" may be individuals or organizations.

  To "modify" a work means to copy from or adapt all or part of the work
in a fashion requiring copyright permission, other than the making of an
exact copy.  The resulting work is called a "modified version" of the
earlier work or a work "based on" the earlier work.

  A "covered work" means either the unmodified Program or a work based
on the Program.

  To "propagate" a work means to do anything with it that, without
permission, would make you directly or secondarily liable for
infringement under applicable copyright law, except executing it on a
computer or modifying a private copy.  Propagation includes copying,
distribution (with or without modification), making available to the
public, and in some countries other activities as well.

  To "convey" a work means any kind of propagation that enables other
parties to make or receive copies.  Mere interaction with a user through
a computer network, with no transfer of a copy, is not conveying.

  An interactive user interface displays "Appropriate Legal Notices"
to the extent that it includes a convenient and prominently visible
feature that (1) displays an appropriate copyright notice, and (2)
tells the user that there is no warranty for the work (except to the
extent that warranties are provided), that licensees may convey the
work under this License, and how to view a copy of this License.  If
the interface presents a list of user commands or options, such as a
menu, a prominent item in the list meets this criterion.

  1. Source Code.

  The "source code" for a work means the preferred form of the work
for making modifications to it.  "Object code" means any non-source
form of a work.

  A "Standard Interface" means an interface that either is an official
standard defined by a recognized standards body, or, in the case of
interfaces specified for a particular programming language, one that
is widely used among developers working in that language.

  The "System Libraries" of an executable work include anything, other
than the work as a whole, that (a) is included in the normal form of
packaging a Major Component, but which is not part of that Major
Component, and (b) serves only to enable use of the work with that
Major Component, or to implement a Standard Interface for which an
implementation is available to the public in source code form.  A
"Major Component", in this context, means a major essential component
(kernel, window system, and so on) of the specific operating system
(if any) on which the executable work runs, or a compiler used to
produce the work, or an object code interpreter used to run it.

  The "Corresponding Source" for a work in object code form means all
the source code needed to generate, install, and (for an executable
work) run the object code and to modify the work, including scripts to
control those activities.  However, it does not include the work's
System Libraries, or general-purpose tools or generally available free
programs which are used unmodified in performing those activities but
which are not part of the work.  For example, Corresponding Source
includes interface definition files associated with source files for
the work, and the source code for shared libraries and dynamically
linked subprograms that the work is specifically designed to require,
such as by intimate data communication or control flow between those
subprograms and other parts of the work.

  The Corresponding Source need not include anything that users
can regenerate automatically from other parts of the Corresponding
Source.

  The Corresponding Source for a work in source code form is that
same work.

  2. Basic Permissions.

  All rights granted under this License are granted for the term of
copyright on the Program, and are irrevocable provided the stated
conditions are met.  This License explicitly affirms your unlimited
permission to run the unmodified Program.  The output from running a
covered work is covered by this License only if the output, given its
content, constitutes a covered work.  This License acknowledges your
rights of fair use or other equivalent, as provided by copyright law.

  You may make, run and propagate covered works that you do not
convey, without conditions so long as your license otherwise remains
in force.  You may convey covered works to others for the sole purpose
of having them make modifications exclusively for you, or provide you
with facilities for running those works, provided that you comply with
the terms of this License in conveying all material for which you do
not control copyright.  Those thus making or running the covered works
for you must do so exclusively on your behalf, under your direction
and control, on terms that prohibit them from making any copies of
your copyrighted material outside their relationship with you.

  Conveying under any other circumstances is permitted solely under
the conditions stated below.  Sublicensing is not allowed; section 10
makes it unnecessary.

  3. Protecting Users' Legal Rights From Anti-Circumvention Law.

  No covered work shall be deemed part of an effective technological
measure under any applicable law fulfilling obligations under article
11 of the WIPO copyright treaty adopted on 20 December 1996, or
similar laws prohibiting or restricting circumvention of such
measures.

  When you convey a covered work, you waive any legal power to forbid
circumvention of technological measures to the extent such circumvention
is effected by exercising rights under this License with respect to
the covered work, and you disclaim any intention to limit operation or
modification of the work as a means of enforcing, against the work's
users, your or third parties' legal rights to forbid circumvention of
technological measures.

  4. Conveying Verbatim Copies.

  You may convey verbatim copies of the Program's source code as you
receive it, in any medium, provided that you conspicuously and
appropriately publish on each copy an appropriate copyright notice;
keep intact all notices stating that this License and any
non-permissive terms added in accord with section 7 apply to the code;
keep intact all notices of the absence of any warranty; and give all
recipients a copy of this License along with the Program.

  You may charge any price or no price for each copy that you convey,
and you may offer support or warranty protection for a fee.

  5. Conveying Modified Source Versions.

  You may convey a work based on the Program, or the modifications to
produce it from the Program, in the form of source code under the
terms of section 4, provided that you also meet all of these conditions:

    a) The work must carry prominent notices stating that you modified
    it, and giving a relevant date.

    b) The work must carry prominent notices stating that it is
    released under this License and any conditions added under section
    7.  This requirement modifies the requirement in section 4 to
    "keep intact all notices".

    c) You must license the entire work, as a whole, under this
    License to anyone who comes into possession of a copy.  This
    License will therefore apply, along with any applicable section 7
    additional terms, to the whole of the work, and all its parts,
    regardless of how they are packaged.  This License gives no
    permission to license the work in any other way, but it does not
    invalidate such permission if you have separately received it.

    d) If the work has interactive user interfaces, each must display
    Appropriate Legal Notices; however, if the Program has interactive
    interfaces that do not display Appropriate Legal Notices, your
    work need not make them do so.

  A compilation of a covered work with other separate and independent
works, which are not by their nature extensions of the covered work,
and which are not combined with it such as to form a larger program,
in or on a volume of a storage or distribution medium, is called an
"aggregate" if the compilation and its resulting copyright are not
used to limit the access or legal rights of the compilation's users
beyond what the individual works permit.  Inclusion of a covered work
in an aggregate does not cause this License to apply to the other
parts of the aggregate.

  6. Conveying Non-Source Forms.

  You may convey a covered work in object code form under the terms
of sections 4 and 5, provided that you also convey the
machine-readable Corresponding Source under the terms of this License,
in one of these ways:

    a) Convey the object code in, or embodied in, a physical product
    (including a physical distribution medium), accompanied by the
    Corresponding Source fixed on a durable physical medium
    customarily used for software interchange.

    b) Convey the object code in, or embodied in, a physical product
    (including a physical distribution medium), accompanied by a
    written offer, valid for at least three years and valid for as
    long as you offer spare parts or customer support for that product
    model, to give anyone who possesses the object code either (1) a
    copy of the Corresponding Source for all the software in the
    product that is covered by this License, on a durable physical
    medium customarily used for software interchange, for a price no
    more than your reasonable cost of physically performing this
    conveying of source, or (2) access to copy the
    Corresponding Source from a network server at no charge.

    c) Convey individual copies of the object code with a copy of the
    written offer to provide the Corresponding Source.  This
    alternative is allowed only occasionally and noncommercially, and
    only if you received the object code with such an offer, in accord
    with subsection 6b.

    d) Convey the object code by offering access from a designated
    place (gratis or for a charge), and offer equivalent access to the
    Corresponding Source in the same way through the same place at no
    further charge.  You need not require recipients to copy the
    Corresponding Source along with the object code.  If the place to
    copy the object code is a network server, the Corresponding Source
    may be on a different server (operated by you or a third party)
    that supports equivalent copying facilities, provided you maintain
    clear directions next to the object code saying where to find the
    Corresponding Source.  Regardless of what server hosts the
    Corresponding Source, you remain obligated to ensure that it is
    available for as long as needed to satisfy these requirements.

    e) Convey the object code using peer-to-peer transmission, provided
    you inform other peers where the object code and Corresponding
    Source of the work are being offered to the general public at no
    charge under subsection 6d.

  A separable portion of the object code, whose source code is excluded
from the Corresponding Source as a System Library, need not be
included in conveying the object code work.

  A "User Product" is either (1) a "consumer product", which means any
tangible personal property which is normally used for personal, family,
or household purposes, or (2) anything designed or sold for incorporation
into a dwelling.  In determining whether a product is a consumer product,
doubtful cases shall be resolved in favor of coverage.  For a particular
product received by a particular user, "normally used" refers to a
typical or common use of that class of product, regardless of the status
of the particular user or of the way in which the particular user
actually uses, or expects or is expected to use, the product.  A product
is a consumer product regardless of whether the product has substantial
commercial, industrial or non-consumer uses, unless such uses represent
the only significant mode of use of the product.

  "Installation Information" for a User Product means any methods,
procedures, authorization keys, or other information required to install
and execute modified versions of a covered work in that User Product from
a modified version of its Corresponding Source.  The information must
suffice to ensure that the continued functioning of the modified object
code is in no case prevented or interfered with solely because
modification has been made.

  If you convey an object code work under this section in, or with, or
specifically for use in, a User Product, and the conveying occurs as
part of a transaction in which the right of possession and use of the
User Product is transferred to the recipient in perpetuity or for a
fixed term (regardless of how the transaction is characterized), the
Corresponding Source conveyed under this section must be accompanied
by the Installation Information.  But this requirement does not apply
if neither you nor any third party retains the ability to install
modified object code on the User Product (for example, the work has
been installed in ROM).

  The requirement to provide Installation Information does not include a
requirement to continue to provide support service, warranty, or updates
for a work that has been modified or installed by the recipient, or for
the User Product in which it has been modified or installed.  Access to a
network may be denied when the modification itself materially and
adversely affects the operation of the network or violates the rules and
protocols for communication across the network.

  Corresponding Source conveyed, and Installation Information provided,
in accord with this section must be in a format that is publicly
documented (and with an implementation available to the public in
source code form), and must require no special password or key for
unpacking, reading or copying.

  7. Additional Terms.

  "Additional permissions" are terms that supplement the terms of this
License by making exceptions from one or more of its conditions.
Additional permissions that are applicable to the entire Program shall
be treated as though they were included in this License, to the extent
that they are valid under applicable law.  If additional permissions
apply only to part of the Program, that part may be used separately
under those permissions, but the entire Program remains governed by
this License without regard to the additional permissions.

  When you convey a copy of a covered work, you may at your option
remove any additional permissions from that copy, or from any part of
it.  (Additional permissions may be written to require their own
removal in certain cases when you modify the work.)  You may place
additional permissions on material, added by you to a covered work,
for which you have or can give appropriate copyright permission.

  Notwithstanding any other provision of this License, for material you
add to a covered work, you may (if authorized by the copyright holders of
that material) supplement the terms of this License with terms:

    a) Disclaiming warranty or limiting liability differently from the
    terms of sections 15 and 16 of this License; or

    b) Requiring preservation of specified reasonable legal notices or
    author attributions in that material or in the Appropriate Legal
    Notices displayed by works containing it; or

    c) Prohibiting misrepresentation of the origin of that material, or
    requiring that modified versions of such material be marked in
    reasonable ways as different from the original version; or

    d) Limiting the use for publicity purposes of names of licensors or
    authors of the material; or

    e) Declining to grant rights under trademark law for use of some
    trade names, trademarks, or service marks; or

    f) Requiring indemnification of licensors and authors of that
    material by anyone who conveys the material (or modified versions of
    it) with contractual assumptions of liability to the recipient, for
    any liability that these contractual assumptions directly impose on
    those licensors and authors.

  All other non-permissive additional terms are considered "further
restrictions" within the meaning of section 10.  If the Program as you
received it, or any part of it, contains a notice stating that it is
governed by this License along with a term that is a further
restriction, you may remove that term.  If a license document contains
a further restriction but permits relicensing or conveying under this
License, you may add to a covered work material governed by the terms
of that license document, provided that the further restriction does
not survive such relicensing or conveying.

  If you add terms to a covered work in accord with this section, you
must place, in the relevant source files, a statement of the
additional terms that apply to those files, or a notice indicating
where to find the applicable terms.

  Additional terms, permissive or non-permissive, may be stated in the
form of a separately written license, or stated as exceptions;
the above requirements apply either way.

  8. Termination.

  You may not propagate or modify a covered work except as expressly
provided under this License.  Any attempt otherwise to propagate or
modify it is void, and will automatically terminate your rights under
this License (including any patent licenses granted under the third
paragraph of section 11).

  However, if you cease all violation of this License, then your
license from a particular copyright holder is reinstated (a)
provisionally, unless and until the copyright holder explicitly and
finally terminates your license, and (b) permanently, if the copyright
holder fails to notify you of the violation by some reasonable means
prior to 60 days after the cessation.

  Moreover, your license from a particular copyright holder is
reinstated permanently if the copyright holder notifies you of the
violation by some reasonable means, this is the first time you have
received notice of violation of this License (for any work) from that
copyright holder, and you cure the violation prior to 30 days after
your receipt of the notice.

  Termination of your rights under this section does not terminate the
licenses of parties who have received copies or rights from you under
this License.  If your rights have been terminated and not permanently
reinstated, you do not qualify to receive new licenses for the same
material under section 10.

  9. Acceptance Not Required for Having Copies.

  You are not required to accept this License in order to receive or
run a copy of the Program.  Ancillary propagation of a covered work
occurring solely as a consequence of using peer-to-peer transmission
to receive a copy likewise does not require acceptance.  However,
nothing other than this License grants you permission to propagate or
modify any covered work.  These actions infringe copyright if you do
not accept this License.  Therefore, by modifying or propagating a
covered work, you indicate your acceptance of this License to do so.

  10. Automatic Licensing of Downstream Recipients.

  Each time you convey a covered work, the recipient automatically
receives a license from the original licensors, to run, modify and
propagate that work, subject to this License.  You are not responsible
for enforcing compliance by third parties with this License.

  An "entity transaction" is a transaction transferring control of an
organization, or substantially all assets of one, or subdividing an
organization, or merging organizations.  If propagation of a covered
work results from an entity transaction, each party to that
transaction who receives a copy of the work also receives whatever
licenses to the work the party's predecessor in interest had or could
give under the previous paragraph, plus a right to possession of the
Corresponding Source of the work from the predecessor in interest, if
the predecessor has it or can get it with reasonable efforts.

  You may not impose any further restrictions on the exercise of the
rights granted or affirmed under this License.  For example, you may
not impose a license fee, royalty, or other charge for exercise of
rights granted under this License, and you may not initiate litigation
(including a cross-claim or counterclaim in a lawsuit) alleging that
any patent claim is infringed by making, using, selling, offering for
sale, or importing the Program or any portion of it.

  11. Patents.

  A "contributor" is a copyright holder who authorizes use under this
License of the Program or a work on which the Program is based.  The
work thus licensed is called the contributor's "contributor version".

  A contributor's "essential patent claims" are all patent claims
owned or controlled by the contributor, whether already acquired or
hereafter acquired, that would be infringed by some manner, permitted
by this License, of making, using, or selling its contributor version,
but do not include claims that would be infringed only as a
consequence of further modification of the contributor version.  For
purposes of this definition, "control" includes the right to grant
patent sublicenses in a manner consistent with the requirements of
this License.

  Each contributor grants you a non-exclusive, worldwide, royalty-free
patent license under the contributor's essential patent claims, to
make, use, sell, offer for sale, import and otherwise run, modify and
propagate the contents of its contributor version.

  In the following three paragraphs, a "patent license" is any express
agreement or commitment, however denominated, not to enforce a patent
(such as an express permission to practice a patent or covenant not to
sue for patent infringement).  To "grant" such a patent license to a
party means to make such an agreement or commitment not to enforce a
patent against the party.

  If you convey a covered work, knowingly relying on a patent license,
and the Corresponding Source of the work is not available for anyone
to copy, free of charge and under the terms of this License, through a
publicly available network server or other readily accessible means,
then you must either (1) cause the Corresponding Source to be so
available, or (2) arrange to deprive yourself of the benefit of the
patent license for this particular work, or (3) arrange, in a manner
consistent with the requirements of this License, to extend the patent
license to downstream recipients.  "Knowingly relying" means you have
actual knowledge that, but for the patent license, your conveying the
covered work in a country, or your recipient's use of the covered work
in a country, would infringe one or more identifiable patents in that
country that you have reason to believe are valid.

  If, pursuant to or in connection with a single transaction or
arrangement, you convey, or propagate by procuring conveyance of, a
covered work, and grant a patent license to some of the parties
receiving the covered work authorizing them to use, propagate, modify
or convey a specific copy of the covered work, then the patent license
you grant is automatically extended to all recipients of the covered
work and works based on it.

  A patent license is "discriminatory" if it does not include within
the scope of its coverage, prohibits the exercise of, or is
conditioned on the non-exercise of one or more of the rights that are
specifically granted under this License.  You may not convey a covered
work if you are a party to an arrangement with a third party that is
in the business of distributing software, under which you make payment
to the third party based on the extent of your activity of conveying
the work, and under which the third party grants, to any of the
parties who would receive the covered work from you, a discriminatory
patent license (a) in connection with copies of the covered work
conveyed by you (or copies made from those copies), or (b) primarily
for and in connection with specific products or compilations that
contain the covered work, unless you entered into that arrangement,
or that patent license was granted, prior to 28 March 2007.

  Nothing in this License shall be construed as excluding or limiting
any implied license or other defenses to infringement that may
otherwise be available to you under applicable patent law.

  12. No Surrender of Others' Freedom.

  If conditions are imposed on you (whether by court order, agreement or
otherwise) that contradict the conditions of this License, they do not
excuse you from the conditions of this License.  If you cannot convey a
covered work so as to satisfy simultaneously your obligations under this
License and any other pertinent obligations, then as a consequence you may
not convey it at all.  For example, if you agree to terms that obligate you
to collect a royalty for further conveying from those to whom you convey
the Program, the only way you could satisfy both those terms and this
License would be to refrain entirely from conveying the Program.

  13. Remote Network Interaction; Use with the GNU General Public License.

  Notwithstanding any other provision of this License, if you modify the
Program, your modified version must prominently offer all users
interacting with it remotely through a computer network (if your version
supports such interaction) an opportunity to receive the Corresponding
Source of your version by providing access to the Corresponding Source
from a network server at no charge, through some standard or customary
means of facilitating copying of software.  This Corresponding Source
shall include the Corresponding Source for any work covered by version 3
of the GNU General Public License that is incorporated pursuant to the
following paragraph.

  Notwithstanding any other provision of this License, you have
permission to link or combine any covered work with a work licensed
under version 3 of the GNU General Public License into a single
combined work, and to convey the resulting work.  The terms of this
License will continue to apply to the part which is the covered work,
but the work with which it is combined will remain governed by version
3 of the GNU General Public License.

  14. Revised Versions of this License.

  The Free Software Foundation may publish revised and/or new versions of
the GNU Affero General Public License from time to time.  Such new versions
will be similar in spirit to the present version, but may differ in detail to
address new problems or concerns.

  Each version is given a distinguishing version number.  If the
Program specifies that a certain numbered version of the GNU Affero General
Public License "or any later version" applies to it, you have the
option of following the terms and conditions either of that numbered
version or of any later version published by the Free Software
Foundation.  If the Program does not specify a version number of the
GNU Affero General Public License, you may choose any version ever published
by the Free Software Foundation.

  If the Program specifies that a proxy can decide which future
versions of the GNU Affero General Public License can be used, that proxy's
public statement of acceptance of a version permanently authorizes you
to choose that version for the Program.

  Later license versions may give you additional or different
permissions.  However, no additional obligations are imposed on any
author or copyright holder as a result of your choosing to follow a
later version.

  15. Disclaimer of Warranty.

  THERE IS NO WARRANTY FOR THE PROGRAM, TO THE EXTENT PERMITTED BY
APPLICABLE LAW.  EXCEPT WHEN OTHERWISE STATED IN WRITING THE COPYRIGHT
HOLDERS AND/OR OTHER PARTIES PROVIDE THE PROGRAM "AS IS" WITHOUT WARRANTY
OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE.  THE ENTIRE RISK AS TO THE QUALITY AND PERFORMANCE OF THE PROGRAM
IS WITH YOU.  SHOULD THE PROGRAM PROVE DEFECTIVE, YOU ASSUME THE COST OF
ALL NECESSARY SERVICING, REPAIR OR CORRECTION.

  16. Limitation of Liability.

  IN NO EVENT UNLESS REQUIRED BY APPLICABLE LAW OR AGREED TO IN WRITING
WILL ANY COPYRIGHT HOLDER, OR ANY OTHER PARTY WHO MODIFIES AND/OR CONVEYS
THE PROGRAM AS PERMITTED ABOVE, BE LIABLE TO YOU FOR DAMAGES, INCLUDING ANY
GENERAL, SPECIAL, INCIDENTAL OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE
USE OR INABILITY TO USE THE PROGRAM (INCLUDING BUT NOT LIMITED TO LOSS OF
DATA OR DATA BEING RENDERED INACCURATE OR LOSSES SUSTAINED BY YOU OR THIRD
PARTIES OR A FAILURE OF THE PROGRAM TO OPERATE WITH ANY OTHER PROGRAMS),
EVEN IF SUCH HOLDER OR OTHER PARTY HAS BEEN ADVISED OF THE POSSIBILITY OF
SUCH DAMAGES.

  17. Interpretation of Sections 15 and 16.

  If the disclaimer of warranty and limitation of liability provided
above cannot be given local legal effect according to their terms,
reviewing courts shall apply local law that most closely approximates
an absolute waiver of all civil liability in connection with the
Program, unless a warranty or assumption of liability accompanies a
copy of the Program in return for a fee.

                     END OF TERMS AND CONDITIONS

            How to Apply These Terms to Your New Programs

  If you develop a new program, and you want it to be of the greatest
possible use to the public, the best way to achieve this is to make it
free software which everyone can redistribute and change under these terms.

  To do so, attach the following notices to the program.  It is safest
to attach them to the start of each source file to most effectively
state the exclusion of warranty; and each file should have at least
the "copyright" line and a pointer to where the full notice is found.

    <one line to give the program's name and a brief idea of what it does.>
    Copyright (C) <year>  <name of author>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

Also add information on how to contact you by electronic and paper mail.

  If your software can interact with users remotely through a computer
network, you should also make sure that it provides a way for users to
get its source.  For example, if your program is a web application, its
interface could display a "Source" link that leads users to an archive
of the code.  There are many ways you could offer source, and different
solutions will be better for different programs; see section 13 for the
specific requirements.

  You should also get your employer (if you work as a programmer) or school,
if any, to sign a "copyright disclaimer" for the program, if necessary.
For more information on this, and how to apply and follow the GNU AGPL, see
<https://www.gnu.org/licenses/>.

```

## `/Users/yuanjing/Code/AutoFilm/Dockerfile`

```
FROM python:3.12.7-alpine

ENV TZ=Asia/Shanghai
VOLUME ["/config", "/logs", "/media","/fonts"]

RUN apk update
RUN apk add --no-cache build-base linux-headers tzdata

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt && \
    rm requirements.txt

COPY app /app

RUN rm -rf /tmp/*

ENTRYPOINT ["python", "/app/main.py"]
```

## `/Users/yuanjing/Code/AutoFilm/.gitignore`

```
config/config.yaml
logs/*.log
strm/*
media/*
__pycache__/*
.mypy_cache/*

.DS_Store
*bak
*test*
*pyc*


```

## `/Users/yuanjing/Code/AutoFilm/.github/workflows/release.yaml`

```
name: AutoFilm Release

on:
  workflow_dispatch:
  push:
    branches:
      - main
    paths:
      - app/version.py

jobs:
  Get-Version:
    runs-on: ubuntu-latest
    outputs:
      app_version: ${{ steps.version.outputs.app_version }}
    steps:
      - name: Clone Repository
        uses: actions/checkout@v4

      - name: APP Version
        id: version
        run: |
          APP_VERSION=$(cat app/version.py | sed -ne 's/APP_VERSION\s=\s"v\(.*\)"/\1/gp')
          echo "检测到版本号为 $APP_VERSION"
          echo "app_version=$APP_VERSION" >> "$GITHUB_OUTPUT"

  Release-Docker-Builder:
    name: Build Docker Image
    needs: [ Get-Version ]
    uses: ./.github/workflows/docker-builder.yaml
    with:
      APP_VERSION: ${{ needs.Get-Version.outputs.app_version }}
      IS_LATEST: true
    secrets:
      DOCKERHUB_USERNAME: ${{ secrets.DOCKERHUB_USERNAME }}
      DOCKERHUB_TOKEN: ${{ secrets.DOCKERHUB_TOKEN }}

  Create-Release:
    permissions: write-all
    runs-on: ubuntu-latest
    needs: [ Get-Version, Release-Docker-Builder ]
    steps:
      - name: Create Release
        id: create_release
        uses: actions/create-release@latest
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        with:
          tag_name: v${{ needs.Get-Version.outputs.app_version }}
          release_name: v${{ needs.Get-Version.outputs.app_version }}
          body: ${{ github.event.commits[0].message }}
          draft: false
          prerelease: false
```

## `/Users/yuanjing/Code/AutoFilm/.github/workflows/docker-builder.yaml`

```
name: AutoFilm Docker Builder

on:
  workflow_call:
    inputs:
      APP_VERSION:
        description: "用于Docker镜像版本的标签号"
        required: true
        type: string
      IS_LATEST:
        description: "是否发布为Docker镜像最新版本"
        required: true
        type: boolean
    secrets:
      DOCKERHUB_USERNAME:
        required: true
      DOCKERHUB_TOKEN:
        required: true

env:
  APP_VERSION: ${{ inputs.APP_VERSION }}
  IS_LATEST: ${{ inputs.IS_LATEST }}

jobs:
  Docker-build:
    runs-on: ubuntu-latest
    name: Build Docker Image
    steps:
      - name: Show Information
        run: |
          echo "Docker镜像版本的标签号：${{ env.APP_VERSION }}"
          echo "是否发布为Docker镜像最新版本：${{ env.IS_LATEST }}"

      - name: Clone Repository
        uses: actions/checkout@v4

      - name: Docker Meta
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ secrets.DOCKERHUB_USERNAME }}/autofilm
          tags: |
              type=raw,value=latest,enable=${{ env.IS_LATEST }}
              type=raw,value=${{ env.APP_VERSION }},enable=true

      - name: Set up QEMU
        uses: docker/setup-qemu-action@v3

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Login to DockerHub
        uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}
        
      - name: Build Image
        uses: docker/build-push-action@v5
        with:
          context: .
          file: ./Dockerfile
          platforms: |
            linux/amd64
            linux/arm64/v8
          push: true
          build-args: |
            AUTOFILM_VERSION=${{ env.APP_VERSION }}
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha, scope=${{ github.workflow }}-docker
          cache-to: type=gha, scope=${{ github.workflow }}-docker
```

## `/Users/yuanjing/Code/AutoFilm/.github/workflows/dev.yaml`

```
name: AutoFilm DEV

on:
  workflow_dispatch:
  push:
    paths:
      - '**.py'

jobs:
  Dev-Docker-Builder:
    name: Build Docker Image
    uses: ./.github/workflows/docker-builder.yaml
    with:
      APP_VERSION: DEV
      IS_LATEST: false
    secrets:
      DOCKERHUB_USERNAME: ${{ secrets.DOCKERHUB_USERNAME }}
      DOCKERHUB_TOKEN: ${{ secrets.DOCKERHUB_TOKEN }}
```
