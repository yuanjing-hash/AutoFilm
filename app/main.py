# /app/main.py (已更新为 lifespan 写法)

from asyncio import get_event_loop
from sys import path
from os.path import dirname
# --- 新增的导入 ---
from contextlib import asynccontextmanager

# FastAPI 和 Uvicorn 的导入
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

path.append(dirname(dirname(__file__)))

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from app.core import settings, logger
from app.extensions import LOGO
from app.modules import Alist2Strm, Ani2Alist, LibraryPoster

# ------------------ 修改部分开始 ------------------

# 存储从配置文件加载的任务配置
CONFIGURED_JOBS = {}

# 使用新的 lifespan 事件处理器
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 应用的生命周期事件管理器
    """
    # === 在应用启动时执行的代码 ===
    global CONFIGURED_JOBS
    scheduler = AsyncIOScheduler()

    logger.info("AutoFilm 启动中...")
    logger.debug(f"是否开启 DEBUG 模式: {settings.DEBUG}")

    if settings.AlistServerList:
        logger.info("检测到 Alist2Strm 模块配置，正在添加至后台任务")
        for server in settings.AlistServerList:
            job_id = server.get("id")
            cron = server.get("cron")
            CONFIGURED_JOBS[f"Alist2Strm_{job_id}"] = {"module": "Alist2Strm", "config": server}
            if cron and job_id:
                scheduler.add_job(
                    Alist2Strm(**server).run, trigger=CronTrigger.from_crontab(cron), id=job_id
                )
                logger.info(f"任务 '{job_id}' (Alist2Strm) 已被添加至后台定时任务")
            else:
                logger.warning(f"任务 {server.get('id', '未知')} 未设置 cron 或 id")
    else:
        logger.warning("未检测到 Alist2Strm 模块配置")

    if settings.Ani2AlistList:
        logger.info("检测到 Ani2Alist 模块配置，正在添加至后台任务")
        for server in settings.Ani2AlistList:
            job_id = server.get("id")
            cron = server.get("cron")
            CONFIGURED_JOBS[f"Ani2Alist_{job_id}"] = {"module": "Ani2Alist", "config": server}
            if cron and job_id:
                scheduler.add_job(
                    Ani2Alist(**server).run, trigger=CronTrigger.from_crontab(cron), id=job_id
                )
                logger.info(f"任务 '{job_id}' (Ani2Alist) 已被添加至后台定时任务")
            else:
                logger.warning(f"任务 {server.get('id', '未知')} 未设置 cron 或 id")
    else:
        logger.warning("未检测到 Ani2Alist 模块配置")

    if settings.LibraryPosterList:
        logger.info("检测到 LibraryPoster 模块配置，正在添加至后台任务")
        for poster in settings.LibraryPosterList:
            job_id = poster.get("id")
            cron = poster.get("cron")
            CONFIGURED_JOBS[f"LibraryPoster_{job_id}"] = {"module": "LibraryPoster", "config": poster}
            if cron and job_id:
                scheduler.add_job(
                    LibraryPoster(**poster).run, trigger=CronTrigger.from_crontab(cron), id=job_id
                )
                logger.info(f"任务 '{job_id}' (LibraryPoster) 已被添加至后台定时任务")
            else:
                logger.warning(f"任务 {poster.get('id', '未知')} 未设置 cron 或 id")
    else:
        logger.warning("未检测到 LibraryPoster 模块配置")

    scheduler.start()
    logger.info("AutoFilm 定时任务启动完成")
    
    yield # yield 之前的代码在启动时运行

    # === 在应用关闭时执行的代码 (如果需要的话) ===
    logger.info("AutoFilm 程序正在关闭...")
    # scheduler.shutdown() # 如果需要优雅关闭调度器可以取消这行注释


# 初始化 FastAPI 应用，并注册 lifespan 事件
app = FastAPI(
    title="AutoFilm API",
    description="一个为 AutoFilm 提供 API 和网页客户端的服务",
    version=settings.APP_VERSION,
    lifespan=lifespan
)

# 定义一个数据模型，用于接收前端传来的任务ID
class JobRequest(BaseModel):
    id: str
    module: str

# ------------------ 修改部分结束 ------------------

def print_logo() -> None:
    """
    打印 Logo
    """
    print(LOGO)
    print(f" {settings.APP_NAME} {settings.APP_VERSION} ".center(65, "="))
    print("")

# 旧的 @app.on_event("startup") 函数已被 lifespan 取代，所以我们把它删除

@app.get("/api/jobs", summary="获取所有已配置的任务列表")
def get_jobs():
    """
    从加载的配置中返回一个清晰的任务列表，供前端使用。
    """
    job_list = []
    # 从 settings 重新加载，保证获取最新配置
    job_list.extend([{"id": j.get("id"), "module": "Alist2Strm", "cron": j.get("cron")} for j in settings.AlistServerList])
    job_list.extend([{"id": j.get("id"), "module": "Ani2Alist", "cron": j.get("cron")} for j in settings.Ani2AlistList])
    job_list.extend([{"id": j.get("id"), "module": "LibraryPoster", "cron": j.get("cron")} for j in settings.LibraryPosterList])
    
    return {"jobs": job_list}


@app.post("/api/trigger_job", summary="立即触发一个指定的后台任务")
async def trigger_job(request: JobRequest):
    """
    根据任务模块和ID，立即执行一次任务。
    这是一个异步操作，会立即返回，任务在后台运行。
    """
    job_key = f"{request.module}_{request.id}"
    job_info = CONFIGURED_JOBS.get(job_key)

    if not job_info:
        raise HTTPException(status_code=404, detail=f"任务 '{request.id}' (模块: {request.module}) 未找到")

    logger.info(f"收到手动触发请求，开始执行任务: '{request.id}'")
    
    module_class = globals()[job_info["module"]]
    config = job_info["config"]

    try:
        # 在后台异步执行任务的 run() 方法
        loop = get_event_loop()
        loop.create_task(module_class(**config).run())
        return {"status": "success", "message": f"任务 '{request.id}' 已在后台开始执行。"}
    except Exception as e:
        logger.error(f"执行任务 '{request.id}' 时出错: {e}")
        raise HTTPException(status_code=500, detail=f"执行任务时出错: {str(e)}")


@app.get("/", response_class=HTMLResponse, summary="提供一个简单的前端控制页面")
async def get_root():
    # 这里我们直接将 HTML 代码返回
    html_content = r"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AutoFilm 控制面板</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; background-color: #f0f2f5; margin: 0; padding: 20px; color: #333; }
            .container { max-width: 800px; margin: 40px auto; background: #fff; padding: 20px 30px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
            h1 { text-align: center; color: #1c1e21; }
            .job-list { list-style: none; padding: 0; }
            .job-item { display: flex; justify-content: space-between; align-items: center; padding: 15px; border-bottom: 1px solid #dddfe2; transition: background-color 0.2s; }
            .job-item:last-child { border-bottom: none; }
            .job-item:hover { background-color: #f7f7f7; }
            .job-details { display: flex; flex-direction: column; }
            .job-id { font-weight: bold; font-size: 1.1em; }
            .job-info { font-size: 0.9em; color: #666; margin-top: 5px; }
            .run-button { background-color: #1877f2; color: white; border: none; padding: 8px 15px; border-radius: 6px; cursor: pointer; font-weight: bold; transition: background-color 0.2s; }
            .run-button:hover { background-color: #166fe5; }
            .run-button:active { background-color: #1462c7; }
            .status { margin-top: 20px; text-align: center; color: green; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>AutoFilm 控制面板</h1>
            <ul id="job-list" class="job-list">
                </ul>
            <p id="status" class="status"></p>
        </div>

        <script>
            const jobListEl = document.getElementById('job-list');
            const statusEl = document.getElementById('status');

            async function fetchJobs() {
                try {
                    const response = await fetch('/api/jobs');
                    if (!response.ok) throw new Error('Failed to fetch jobs');
                    const data = await response.json();
                    
                    jobListEl.innerHTML = ''; // Clear list
                    if (data.jobs && data.jobs.length > 0) {
                        data.jobs.forEach(job => {
                            const li = document.createElement('li');
                            li.className = 'job-item';
                            li.innerHTML = `
                                <div class="job-details">
                                    <span class="job-id">${job.id}</span>
                                    <span class="job-info">模块: ${job.module} | 计划: ${job.cron || '无'}</span>
                                </div>
                                <button class="run-button" onclick="triggerJob('${job.id}', '${job.module}')">立即运行</button>
                            `;
                            jobListEl.appendChild(li);
                        });
                    } else {
                        jobListEl.innerHTML = '<p>在 config.yaml 中没有找到任何可执行的任务。</p>';
                    }
                } catch (error) {
                    jobListEl.innerHTML = `<p style="color: red;">加载任务列表失败: ${error.message}</p>`;
                }
            }

            async function triggerJob(jobId, jobModule) {
                statusEl.textContent = `正在触发任务: ${jobId}...`;
                try {
                    const response = await fetch('/api/trigger_job', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ id: jobId, module: jobModule }),
                    });
                    const result = await response.json();
                    if (!response.ok) {
                        throw new Error(result.detail || 'Unknown error');
                    }
                    statusEl.textContent = result.message;
                } catch (error) {
                    statusEl.textContent = `触发失败: ${error.message}`;
                    statusEl.style.color = 'red';
                }
                setTimeout(() => { 
                    statusEl.textContent = '';
                    statusEl.style.color = 'green';
                }, 5000);
            }

            // Load jobs on page load
            window.onload = fetchJobs;
        </script>
    </body>
    </html>
    """;
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    print_logo()

    # Uvicorn 会负责启动应用和管理事件循环，它会自动调用 lifespan 函数
    uvicorn.run(
        "main:app",  # 指向 main.py 文件中的 app 对象
        host="0.0.0.0",
        port=8000,
        reload=False # 在生产环境中应设为 False
    )