from django.shortcuts import render
from .models import Path


def upload_data(request):
    if request.method == 'POST':
        data_dir = request.POST.get('data_dir')
        print(data_dir)
        Path.objects.get_or_create(path=data_dir)
        return plots(request, data_dir)
    else:
        return render(request, 'core/upload_data.html', {})


def data_for_plots(data_dir):
    pass


def plots(request, data_dir):
    data = data_for_plots(data_dir)
    return render(request, 'core/plots.html', {data})
