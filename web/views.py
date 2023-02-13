from django.shortcuts import render
from django.http import StreamingHttpResponse,JsonResponse
from slrsite.camera import gen,getAns
from slrsite.actionModel import getAll,find

def Index(request):
    return render(request,'index.html')

def About(request):
    return render(request,'about.html')

def Team(request):
    return render(request,'team.html')

def Ai(request):
    return render(request,'ai.html')

def Action(request):
    action=getAll()
    return render(request,'action.html',{'action':action})

def Search(request):
        # create a form instance and populate it with data from the request:

    #if request.method=='POST':
    search=request.GET['search']
    action=find(search)
    return render(request,'search.html',{'action':action})
    #return render(request,'search.html')
def get_data(request):
    data = getAns()
    return JsonResponse({'data':data})

def video_stream(request):
    return StreamingHttpResponse(gen(),content_type='multipart/x-mixed-replace; boundary=frame')


