import torch
import warnings


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class _ECELoss(torch.nn.Module):
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

        bin_boundaries_plot = torch.linspace(0, 1, 11)
        self.bin_lowers_plot = bin_boundaries_plot[:-1]
        self.bin_uppers_plot = bin_boundaries_plot[1:]

    def forward(self, confidences, predictions, labels, title=None):
        accuracies = predictions.eq(labels)
        ece = torch.zeros(1, device=confidences.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        accuracy_in_bin_list = []
        for bin_lower, bin_upper in zip(self.bin_lowers_plot, self.bin_uppers_plot):
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            accuracy_in_bin = 0
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean().item()
            accuracy_in_bin_list.append(accuracy_in_bin)

        if title:
            fig = plt.figure(figsize=(8, 6))
            p1 = plt.bar(np.arange(10) / 10., accuracy_in_bin_list, 0.1, align='edge', edgecolor='black')
            p2 = plt.plot([0, 1], [0, 1], '--', color='gray')

            plt.ylabel('Accuracy', fontsize=18)
            plt.xlabel('Confidence', fontsize=18)
            # plt.title(title)
            plt.xticks(np.arange(0, 1.01, 0.2), fontsize=12)
            plt.yticks(np.arange(0, 1.01, 0.2), fontsize=12)
            plt.xlim(left=0, right=1)
            plt.ylim(bottom=0, top=1)
            plt.grid(True)
            # plt.legend((p1[0], p2[0]), ('Men', 'Women'))
            plt.text(0.1, 0.83, 'ECE: {:.4f}'.format(ece.item()), fontsize=18)
            fig.tight_layout()
            plt.savefig(title, format='pdf', dpi=600, bbox_inches='tight')

        return ece


def psd_safe_eigen(K):
    Kprime = K.clone()
    jitter = 0
    jitter_new = None
    while True:
        p, q = torch.linalg.eigh(Kprime)
        if (p > 0).all():
            if jitter_new is not None:
                warnings.warn(
                    f"K not p.d., added jitter of {jitter_new} to the diagonal",
                    RuntimeWarning,
                )
            return p, q
        else:
            if jitter == 0:
                jitter_new = 1e-10
            else:
                jitter_new = jitter * 10
        Kprime.diagonal().add_(jitter_new - jitter)
        jitter = jitter_new


def psd_safe_cholesky(A, upper=False, out=None, jitter=None):
    """Compute the Cholesky decomposition of A. If A is only p.s.d, add a small jitter to the diagonal.
    Args:
        :attr:`A` (Tensor):
            The tensor to compute the Cholesky decomposition of
        :attr:`upper` (bool, optional):
            See torch.cholesky
        :attr:`out` (Tensor, optional):
            See torch.cholesky
        :attr:`jitter` (float, optional):
            The jitter to add to the diagonal of A in case A is only p.s.d. If omitted, chosen
            as 1e-6 (float) or 1e-8 (double)
    """
    try:
        L = torch.linalg.cholesky(A, upper=upper, out=out)
        return L
    except RuntimeError as e:
        isnan = torch.isnan(A)
        if isnan.any():
            raise NanError(
                f"cholesky_cpu: {isnan.sum().item()} of {A.numel()} elements of the {A.shape} tensor are NaN."
            )

        if jitter is None:
            jitter = 1e-6 if A.dtype == torch.float32 else 1e-8
        Aprime = A.clone()
        jitter_prev = 0
        for i in range(5):
            jitter_new = jitter * (10 ** i)
            Aprime.diagonal(dim1=-2, dim2=-1).add_(jitter_new - jitter_prev)
            jitter_prev = jitter_new
            try:
                L = torch.linalg.cholesky(Aprime, upper=upper, out=out)
                warnings.warn(
                    f"A not p.d., added jitter of {jitter_new} to the diagonal",
                    RuntimeWarning,
                )
                return L
            except RuntimeError:
                continue
        raise e


def find_module_by_name(model, name):
    names = name.split(".")
    module = model
    for n in names[:-1]:
        module = getattr(module, n)
    return module, names[-1]