fig = plt.figure(figsize=(15,14))
i = 1
ansatz = 'Weighted_2D'
for function in ['Himmelblau','Brent','Adjiman','Threehump']:
    ax = fig.add_subplot(2, 2, i, projection='3d')
    paint_real_2D(function.lower(), ansatz, ax, df, L)
    pos = ax.get_position()
    pos.x0 -= 0.05
    pos.y1 += 0.05
    if i > 2:
        pos.y1 -= 0.02
        pos.y0 -= 0.02
    ax.set_position(pos)
    ax.set_xlabel('x', fontsize=20)
    ax.set_ylabel('y', fontsize=20)
    ax.set_zlabel('z', fontsize=20)
    ax.set_title(function, fontsize=20)
    ax.set_xticks([-5,0,5])
    ax.set_yticks([-5, 0, 5])
    ax.set_zticks([-1,0,1])
    ax.tick_params(axis='both', which='major', labelsize=18)
    i += 1

handles = []

handles.append(mlines.Line2D([], [], color=colors['classical'], markersize=10, label='Classical' , linewidth=0, marker=marker['classical']))
handles.append(mlines.Line2D([], [], color=colors['quantum'], markersize=10, label='Quantum' ,
                             linewidth=0, marker=marker['quantum']))
handles.append(mlines.Line2D([], [], color=colors['experiment'], markersize=10, label='Experiment' ,
                             linewidth=0, marker=marker['experiment']))
fig.legend(handles = handles, bbox_to_anchor=(0.4, -0.07, 0.18, .2),borderaxespad=0., mode='expand', fontsize=20, ncol=1)

#fig.text(0.35, 0.04, '%s layers'%L, fontsize=20, fontweight='bold')
fig.savefig('functions_2D_%sL.pdf'%L)