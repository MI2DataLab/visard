import itertools

from django.shortcuts import render, get_object_or_404
import pandas as pd

from .models import Path


def upload_data(request):
    if request.method == 'POST':
        data_dir = request.POST.get('data_dir')
        print(data_dir)
        path = Path.objects.get_or_create(path=data_dir)
        return plots(request, path.pk)
    else:
        return render(request, 'core/upload_data.html', {})


def data_for_plots(data_dir):
    data = pd.read_csv(data_dir)
    sorted_cols = sort_cols(data)
    sorted_pairs = sort_pairs(data)
    return data, sorted_cols, sorted_pairs


def sort_cols(data):
    return data.columns


def sort_pairs(data):
    return list(itertools.combinations(data.columns, 2))


def plots(request, path_id):
    path = get_object_or_404(Path, pk=path_id)
    data, sorted_cols, sorted_pairs = data_for_plots(path.path)
    return render(
        request,
        'core/plots.html',
        {'data': data, 'sorted_cols': sorted_cols, 'sorted_pairs': sorted_pairs},
    )
