from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO
import base64

from fastai import *
from fastai.vision import *
from fastai.callbacks.hooks import *

model_file_url = 'https://www.dropbox.com/s/w9oztw3j7ndbivf/model.pth?raw=1'
model_file_name = 'model'
classes = ['affenpinscher','afghan_hound','african_hunting_dog','airedale','american_staffordshire_terrier','appenzeller','australian_terrier','basenji','basset','beagle','bedlington_terrier','bernese_mountain_dog','black-and-tan_coonhound','blenheim_spaniel','bloodhound','bluetick','border_collie','border_terrier','borzoi','boston_bull','bouvier_des_flandres','boxer','brabancon_griffon','briard','brittany_spaniel','bull_mastiff','cairn','cardigan','chesapeake_bay_retriever','chihuahua','chow','clumber','cocker_spaniel','collie','curly-coated_retriever','dandie_dinmont','dhole','dingo','doberman','english_foxhound','english_setter','english_springer','entlebucher','eskimo_dog','flat-coated_retriever','french_bulldog','german_shepherd','german_short-haired_pointer','giant_schnauzer','golden_retriever','gordon_setter','great_dane','great_pyrenees','greater_swiss_mountain_dog','groenendael','ibizan_hound','irish_setter','irish_terrier','irish_water_spaniel','irish_wolfhound','italian_greyhound','japanese_spaniel','keeshond','kelpie','kerry_blue_terrier','komondor','kuvasz','labrador_retriever','lakeland_terrier','leonberg','lhasa','malamute','malinois','maltese_dog','mexican_hairless','miniature_pinscher','miniature_poodle','miniature_schnauzer','newfoundland','norfolk_terrier','norwegian_elkhound','norwich_terrier','old_english_sheepdog','otterhound','papillon','pekinese','pembroke','pomeranian','pug','redbone','rhodesian_ridgeback','rottweiler','saint_bernard','saluki','samoyed','schipperke','scotch_terrier','scottish_deerhound','sealyham_terrier','shetland_sheepdog','shih-tzu','siberian_husky','silky_terrier','soft-coated_wheaten_terrier','staffordshire_bullterrier','standard_poodle','standard_schnauzer','sussex_spaniel','tibetan_mastiff','tibetan_terrier','toy_poodle','toy_terrier','vizsla','walker_hound','weimaraner','welsh_springer_spaniel','west_highland_white_terrier','whippet','wire-haired_fox_terrier','yorkshire_terrier']
path = Path(__file__).parent
data_bunch = ImageDataBunch.single_from_classes(path, classes, tfms=get_transforms(), size=224).normalize(imagenet_stats)

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)

async def setup_learner():
    await download_file(model_file_url, path/'models'/f'{model_file_name}.pth')
    learn = create_cnn(data_bunch, models.resnet34, metrics=error_rate, bn_final=True)
    learn.load(model_file_name)
    return learn

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

@app.route('/')
def index(request):
    html = path/'view'/'index.html'
    return HTMLResponse(html.open().read())

@app.route('/analyze', methods=['POST'])
async def analyze(request):
    data = await request.body()
    img = open_image(BytesIO(data))
    pred_class,pred_idx,outputs = learn.predict(img)
    # Create heat map 
    xb,_ = data_bunch.one_item(img)
    xb_im = Image(data_bunch.denorm(xb)[0])
    hook_a,hook_g = hooked_backward(learn, pred_idx, xb)
    acts  = hook_a.stored[0]
    avg_acts = acts.mean(0)
    print(avg_acts.shape)
    fig,ax = plt.subplots()
    xb_im.show(ax)
    ax.imshow(avg_acts, alpha=0.8, extent=(0,352,352,0), interpolation='bilinear', cmap='hot')
    plt.savefig('heat.png')
    return JSONResponse({'result': classes[pred_idx], 'heatmap': ''})

def hooked_backward(learner, cat, xb):
    m = learner.model.eval();
    with hook_output(m[0]) as hook_a: 
        with hook_output(m[0], grad=True) as hook_g:
            preds = m(xb)
            preds[0,int(cat)].backward()
    return hook_a, hook_g

if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app, host='0.0.0.0', port=5042)

